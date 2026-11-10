"""Does this machine's Playwright Chromium expose a usable WebGPU adapter?  uv run python .../webgpu_probe.py"""
from playwright.sync_api import sync_playwright

JS = """async () => {
  if (!navigator.gpu) return {gpu: false};
  const a = await navigator.gpu.requestAdapter();
  if (!a) return {gpu: true, adapter: null};
  const info = a.info || {};
  return {gpu: true, adapter: {vendor: info.vendor, architecture: info.architecture, description: info.description,
          f16: a.features.has('shader-f16'), maxBuffer: a.limits.maxBufferSize}};
}"""
for args in ([], ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan"]):
    with sync_playwright() as p:
        b = p.chromium.launch(args=args)
        pg = b.new_page(); pg.goto("about:blank")
        print(args or "default", pg.evaluate(JS))
        b.close()
