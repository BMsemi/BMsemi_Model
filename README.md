# BMsemi Model Library

Canonical source of model adapters, weights, reference projects and hardware assets for the Neuromorphic Platform.

## Contents
- `catalog.json`: runnable built-in operations and module names.
- `bm_models/`: platform-compatible software adapters and mapping code.
- `projects/`: original foundation, building-block and example-model projects, including Deep Gate, MNIST, TCAM, YOLO26 and FP4LLM/Qwen.
- `hardware/`: ZCU106 graph runner. The current runner supports only the PE self-test; archived model code is not automatically FPGA-compatible.
- `templates/simple-model/`: minimal contribution package.

## Contribute through the platform
Create a GitHub repository containing `model.json` and `model.py` at its root, following the template. Sign into the platform, open Models, import that repository, and submit a contribution. The platform publishes a contribution branch here. An administrator reviews the files and publishes the exact submitted commit. Only published main-branch code is activated for canvas and API users.

`run(params)` must accept a JSON object and return a JSON-serializable dictionary. New contributions belong to the `community` module; an administrator grants that module to users. No dependency installation or source code execution occurs during import. Dependencies must already exist on the platform; request administrator installation when needed. Contributions are reviewed Python code running with the platform process permissions, not a sandbox.

## Licensing
Original copyright and license notices are retained in project files. No blanket license or new rights over third-party models are implied. Contributors must have permission to publish their submitted code, data and weights. Do not submit passwords, access tokens, personal datasets or environment files.

## Runtime provenance
The platform uses a commit-pinned checkout of this repository. Git fetch/import never changes active code until publication or administrator sync. Built-in model adapters use assets within that checkout; local project directories are not a fallback.

## Model inputs: canvas and APIs

| Model / operation | User input | Result |
|---|---|---|
| YOLO / `models.yolo26.detect` | One PNG/JPEG image, <=5 MB | Object labels, confidence, boxes; optional annotated JPEG |
| TCAM_Search_FPGA / `models.tcam.search` | One PNG/JPEG image, <=5 MB, plus 1-32 search entries (`key`, `chunk_id`, optional `care`) | 16-bit image fingerprint, ranked matches, `hit`, matching `chunk_id` |
| Deepgate / `models.deepgate.infer` | Dataset of 1-20 WAV recordings, mono/stereo, 8-96 kHz, <=30 seconds and <=10 MB each, <=20 MB total | Per-recording reconstruction MSE; `is_anomaly` only when a calibrated threshold is supplied |
| MNIST / `models.mnist.edge_filter` | TXT/CSV: exactly 28 rows x 28 numeric pixels; any digit 0-9 | Edge map and summary; this adapter does not predict the digit class |

### Canvas

Drag a model onto the canvas and select it. The input panel explains its format and limits. Choose the file(s); for MNIST you may paste the pixel grid instead. Set **Software simulator**, edit optional JSON settings, and click **Run graph**. Select **Show upload API example** to copy a Python example for that model. Settings also lists each model's input contract and upload endpoint. Large attachments may need to be reattached after a page refresh.

**Hardware scope:** the current YOLO, TCAM, Deepgate and MNIST adapters execute in software. The TCAM source project is named TCAM_Search_FPGA, but its current platform adapter uses a software PE emulator. The physical FPGA graph runner currently accepts only `core.pe_self_test`.

### Input details

- **YOLO:** tune `conf`, `iou`, `max_detections`; set `return_image: true` to include `annotated_image_base64` in JSON. No training dataset is required for one-image inference.
- **TCAM:** the image is converted to a 16-bit visual fingerprint. Supply your search table in `entries`, for example `[{"key":"0x36C8","chunk_id":1}]`. This is only a format example, not a meaningful image database. `top_k` controls returned candidates; `max_hamming` controls the hit threshold. A nonmatching image can validly produce `hit: false` and `chunk_id: null`. In the JSON API, provide exactly one of `image_base64` or the advanced numeric `query`, never both.
- **Deepgate:** only WAV is supported by the new upload path, not MP3, ZIP, or a folder upload. Select multiple WAV files as a dataset. Stereo channels are averaged; the original sample rate is retained. Preprocessing follows the repository's MFCC extraction: 20 coefficients, FFT 2048, hop 512, 128 mel bands, first 32 frames, zero-padded if shorter. Remaining frames are not scored. The 640 MFCC values are quantized and passed to the existing INT8 autoencoder. No spectral noise subtraction is applied. Scores are feature reconstruction errors, not transcription, audio classification or denoised audio. Determine `threshold` from representative normal/anomalous validation recordings. Advanced JSON inputs (`mfcc`, `input`, or built-in `case`) remain available.
- **MNIST:** 28 nonempty rows, each with 28 space- or comma-separated integer pixels. Binary values use 0=background and 1=foreground. Grayscale 0-255 is converted to binary using >=128 as foreground. The file represents the handwritten digit, not a single number such as `7`. Any digit 0-9 can be supplied this way; only demo samples 5 and 9 are bundled. Optional `output: "summary"` omits the full edge map.

### Multipart API (no manual base64)

Use `POST /v1/upload/<operation>` with `X-API-Key`, repeated multipart field **files**, and field **params** containing a JSON object. The key needs the corresponding module (`yolo26`, `tcam`, `deepgate`, or `mnist`). This endpoint returns `operation`, `backend: "software"`, and `result`.

```python
import json, os, requests
from contextlib import ExitStack

# Change operation, paths and params together for the desired model.
operation = "models.deepgate.infer"
paths = ["recording1.wav", "recording2.wav"]
params = {"threshold": None, "layers": False}
with ExitStack() as stack:
    files = [("files", (path, stack.enter_context(open(path, "rb"))))
             for path in paths]
    r = requests.post(
        "https://neuro.bmsemiconnect.com/v1/upload/" + operation,
        headers={"X-API-Key": os.environ["NEURO_API_KEY"]},
        data={"params": json.dumps(params)}, files=files, timeout=120)
r.raise_for_status()
print(r.json())
```

For YOLO use `paths=["photo.jpg"]`, `operation="models.yolo26.detect"`, and `params={"return_image":True}`. For TCAM use an image with `operation="models.tcam.search"` and `params={"entries":[{"key":"0x36C8","chunk_id":1}],"top_k":4,"max_hamming":0}`. For MNIST use `paths=["digit.txt"]`, `operation="models.mnist.edge_filter"`, and `params={"output":"map"}`.

### JSON API and graph inputs

Existing `POST /v1/run/<operation>` accepts `{"params": {...}}`. Images use `image_base64`; Deepgate uses `audio_dataset: [{"name":"a.wav","audio_base64":"..."}]` or a single `audio_base64`; MNIST uses `text_data` containing the grid string. Supply actual base64 file contents, without a data-URL prefix. Use these same fields inside each node's `params` for JSON graph API calls. The existing multipart graph API binds only `$image`; audio/text graphs should use JSON node parameters. Uploaded audio takes precedence over advanced Deepgate reference inputs; keep only the intended input type.

Invalid dimensions, malformed data and incompatible formats produce HTTP 422; oversized multipart files produce HTTP 413. Missing model dependencies produce HTTP 503. Empty datasets and a bare digit label are not valid uploads.

## PE behavioral simulator

The updated Neuromorphic_X1 kanna PE implementation is available as core.pe_simulate (software). See [PE simulator inputs, cycles and energy](docs/PE_SIMULATOR.md). The physical ZCU106 self-test remains core.pe_self_test.

