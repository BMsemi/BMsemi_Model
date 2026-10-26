# Community Double test model

Software arithmetic demo; no trained weights or FPGA execution.

Canvas: refresh, drag Community models / community.double onto the canvas,
choose Software simulator, set {"value": 2}, and Run graph.
Expected result: {"value": 4.0}. Try -3 to get -6.0.

API: POST https://neuro.bmsemiconnect.com/v1/run/community.double
Header: X-API-Key with a key enabled for the community module.
JSON body: {"params": {"value": 2}}
The response result field contains {"value": 4.0}.
Non-admin users need their admin to assign the community module.

value accepts finite JSON numbers and defaults to 2. Strings, booleans,
and non-finite or overflowing results are rejected.

To reuse this contribution, copy model.json and model.py to your repository
root, choose a unique dotted name in the manifest, and import your repository
through the platform Models tab.
