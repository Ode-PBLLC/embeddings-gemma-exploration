import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";
const [base, ...dtypes] = process.argv.slice(2);
const b = await chromium.launch({ channel: "msedge", headless: true, args: ["--enable-unsafe-webgpu"] });
for (const spec of dtypes) {
  const [dtype, device] = spec.split("@");
  const pg = await b.newPage();
  pg.on("pageerror", (e) => console.log("pageerror", String(e)));
  await pg.goto(`${base}/index.html?dtype=${dtype}&device=${device}&all=1`);
  if (spec === dtypes[0]) console.log("adapter", JSON.stringify(await pg.evaluate(async () => { const a = navigator.gpu && await navigator.gpu.requestAdapter(); return a ? { ...a.info && { vendor: a.info.vendor, arch: a.info.architecture }, f16: a.features.has("shader-f16") } : null; })));
  await pg.waitForFunction(() => window.RESULT, null, { timeout: 900000 });
  const r = await pg.evaluate(() => window.RESULT); if (r.vectors) writeFileSync(`browser_${dtype}_${device}.json`, JSON.stringify(r.vectors)); delete r.vectors; delete r.query_ms; console.log(JSON.stringify(r));
  await pg.close();
}
await b.close();
