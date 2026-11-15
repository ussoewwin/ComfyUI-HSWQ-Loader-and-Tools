import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "HSWQLoadIPAdapterModel";
const LIST_URL = "/hswq/ipadapter/list";

/**
 * Ask the backend which model files live in `dir`.
 * An empty dir means the ComfyUI-registered models/ipadapter folder.
 */
async function listFiles(dir) {
    const url = LIST_URL + (dir ? `?dir=${encodeURIComponent(dir)}` : "");
    const res = await api.fetchApi(url);
    const data = await res.json();
    return Array.isArray(data) ? data : [];
}

/**
 * Keep the native "ipadapter_file" combo in sync with the "ipadapter_dir"
 * text field: on a folder change, replace the combo's option list with the
 * files of that folder and keep the selection valid.
 *
 * The backend declares the file input as a COMBO, so this widget already is
 * a dropdown - no widget replacement happens here. The backend accepts any
 * file of any folder because it validates the input itself (VALIDATE_INPUTS).
 */
function setupFileCombo(node) {
    const dirWidget = node.widgets?.find((w) => w.name === "ipadapter_dir");
    const fileWidget = node.widgets?.find((w) => w.name === "ipadapter_file");
    if (!dirWidget || !fileWidget || fileWidget.type !== "combo") return null;

    let refreshSeq = 0;
    const refresh = async (dirValue) => {
        const seq = ++refreshSeq;
        const dir = (dirValue ?? dirWidget.value ?? "").trim();
        let files = [];
        try {
            files = await listFiles(dir);
        } catch (err) {
            console.warn("[HSWQ IPAdapter] file listing failed", err);
            return;
        }
        if (seq !== refreshSeq) return; // a newer change already ran
        if (!files.length) files = [""];

        fileWidget.options.values = files;
        if (!files.includes(fileWidget.value)) {
            fileWidget.value = files[0];
            fileWidget.callback?.(fileWidget.value);
        }
        node.setDirtyCanvas?.(true, false);
        node.graph?.incrementVersion?.();
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

    refresh(dirWidget.value);
    return fileWidget;
}

function applySetup(node) {
    if (!node || node.comfyClass !== NODE_ID) return;
    if (node.__hswqIpAdapterComboReady) return;
    const combo = setupFileCombo(node);
    if (!combo) return;
    node.__hswqIpAdapterComboReady = true;
}

app.registerExtension({
    name: "hswq.ipadapter_model_dir",

    nodeCreated(node) {
        applySetup(node);
        setTimeout(() => applySetup(node), 100);
    },

    loadedGraphNode(node) {
        applySetup(node);
        setTimeout(() => applySetup(node), 100);
    },
});
