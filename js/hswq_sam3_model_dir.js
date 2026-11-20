import { app as shimApp } from "../../scripts/app.js";
import { api as shimApi } from "../../scripts/api.js";

const NODE_IDS = ["HSWQLoadConvRotINT8SAM3", "HSWQSAM3Loader"];
// Shared with the other HSWQ folder-combo extension: only one of them binds
// a node, so a node is never driven by two listings at once.
const BOUND_FLAG = "__hswqDirComboBound";
const LIST_URL = "/hswq/sam3/list";

// The legacy shims export window.comfyAPI.*; read them directly as a fallback
// so the extension keeps working if the shim import resolves to undefined.
const app = shimApp ?? globalThis?.comfyAPI?.app?.app;
const api = shimApi ?? globalThis?.comfyAPI?.api?.api;

/**
 * Ask the backend which model files live in `dir`.
 * An empty dir means the registered SAM3 model folders.
 */
async function listFiles(dir) {
    const url = LIST_URL + (dir ? `?dir=${encodeURIComponent(dir)}` : "");
    const res = await api.fetchApi(url);
    const data = await res.json();
    return Array.isArray(data) ? data : [];
}

/**
 * Replace the dropdown options while keeping the array reactive.
 * The frontend store keeps `options` reactive; mutating the existing array is
 * what its own combo code does (`values.splice(...)`), so do the same.
 */
function setComboValues(widget, values) {
    const current = widget.options?.values;
    if (Array.isArray(current)) {
        current.splice(0, current.length, ...values);
        return;
    }
    widget.options = { ...(widget.options ?? {}), values };
}

/**
 * The frontend's "missing model" scan compares the widget's value against the
 * widget's own option list. A workflow that points at a folder outside the
 * registered model folders is therefore reported as missing until the option
 * list of that folder has been filled in - and the scan has to run again
 * afterwards, otherwise the state stays stale.
 */
let rescanTimer = null;
let rescanInFlight = false;
function scheduleMissingRescan() {
    if (rescanTimer) clearTimeout(rescanTimer);
    rescanTimer = setTimeout(async () => {
        rescanTimer = null;
        if (rescanInFlight) return;
        rescanInFlight = true;
        try {
            // reloadDefs: false - the option list of this widget is already
            // the folder's list; reloading the node definitions would reset it.
            await app.refreshMissingModels?.({ silent: true, reloadDefs: false });
        } catch (err) {
            console.warn("[HSWQ SAM3] missing-model rescan failed", err);
        } finally {
            setTimeout(() => {
                rescanInFlight = false;
            }, 250);
        }
    }, 100);
}

/**
 * Keep the native "sam3_name" dropdown in sync with the "sam3_dir" folder
 * field: on a folder change, replace the dropdown's option list with the
 * files of that folder.
 *
 * The selected file name is never replaced by the listing. A saved workflow
 * keeps the file name that belongs to the folder the user typed, and the
 * listing may still belong to the previous folder while the folder widget is
 * being restored.
 */
function setupFileCombo(node) {
    if (!node?.widgets) return null;
    const dirWidget = node.widgets.find((w) => w.name === "sam3_dir");
    const fileWidget = node.widgets.find((w) => w.name === "sam3_name");
    if (!dirWidget || !fileWidget || fileWidget.type !== "combo") return null;

    let seq = 0;
    const refresh = async (dirValue) => {
        const current = ++seq;
        const dir = String(dirValue ?? dirWidget.value ?? "").trim();
        let files = [];
        try {
            files = await listFiles(dir);
        } catch (err) {
            console.warn("[HSWQ SAM3] file listing failed", err);
            return;
        }
        if (current !== seq) return; // a newer change already ran

        const selected = String(fileWidget.value ?? "");
        const values = files.slice();
        if (selected && !values.includes(selected)) values.unshift(selected);
        if (!values.length) values.push("");
        setComboValues(fileWidget, values);

        node.setDirtyCanvas?.(true, false);
        scheduleMissingRescan();
    };

    if (!dirWidget.__hswqWrapped) {
        const orig = dirWidget.callback;
        dirWidget.callback = function (value, ...rest) {
            const result = orig?.call(this, value, ...rest);
            refresh(typeof value === "string" ? value : dirWidget.value);
            return result;
        };
        dirWidget.__hswqWrapped = true;
    }

    if (!node.__hswqChangedWrapped) {
        const origChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, oldValue, widget) {
            const result = origChanged?.apply(this, arguments);
            if (name === "sam3_dir") {
                refresh(typeof value === "string" ? value : dirWidget.value);
            }
            return result;
        };
        node.__hswqChangedWrapped = true;
    }

    // Entry point used after a saved graph is restored: read the folder the
    // workflow restored, never the value from node creation time.
    node.__hswqRefreshCombo = () => refresh(dirWidget.value);

    refresh(dirWidget.value);
    // A saved workflow restores widget values after node creation, and the
    // frontend may reload node definitions afterwards; re-list a few times so
    // the dropdown ends up on the restored folder. The selection is untouched.
    for (const delay of [50, 250, 700, 1500, 3000]) {
        setTimeout(() => refresh(dirWidget.value), delay);
    }
    return fileWidget;
}

function refreshNode(node) {
    const refresh = node?.__hswqRefreshCombo;
    if (typeof refresh === "function") refresh();
}

function applySetup(node) {
    if (!node || !NODE_IDS.includes(node.comfyClass)) return;
    if (node[BOUND_FLAG]) return;
    let attempts = 0;
    const trySetup = () => {
        if (node[BOUND_FLAG]) return;
        const combo = setupFileCombo(node);
        if (combo) {
            node[BOUND_FLAG] = true;
            return;
        }
        // The frontend builds the node's widgets asynchronously; keep waiting
        // until the folder field and the dropdown both exist.
        if (++attempts < 60) setTimeout(trySetup, 100);
    };
    trySetup();

    // A saved workflow fills the widgets in onConfigure; list the folder the
    // workflow restored at that point, so the file it points at is an option
    // again right after a start.
    if (!node.__hswqConfigureWrapped) {
        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            const result = origConfigure?.apply(this, arguments);
            setTimeout(() => refreshNode(this), 0);
            return result;
        };
        node.__hswqConfigureWrapped = true;
    }
}

if (app?.registerExtension) {
    app.registerExtension({
        name: "hswq.sam3_model_dir",

        nodeCreated(node) {
            applySetup(node);
        },

        loadedGraphNode(node) {
            applySetup(node);
            if (!node || !NODE_IDS.includes(node.comfyClass)) return;
            // Runs after the saved widget values are restored.
            refreshNode(node);
        },

        // Called by the frontend after node definitions are (re)loaded; the
        // combo options are reset from the node definition at that point.
        refreshComboInNodes() {
            const graph = app.graph;
            for (const node of graph?._nodes ?? []) {
                if (!NODE_IDS.includes(node?.comfyClass)) continue;
                refreshNode(node);
                node[BOUND_FLAG] = false;
                applySetup(node);
            }
        },
    });
} else {
    console.error("[HSWQ SAM3] app.registerExtension is unavailable; the folder dropdown cannot follow sam3_dir");
}
