import { launch } from "puppeteer";
import { writeFile } from "node:fs";
import { randomUUID } from "node:crypto";
const getImage = async () => {
  const browser = await launch({
    browser: "chrome",
    headless: false,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-sync",
      "--ignore-certificate-errors",
    ],
  });
  const page = await browser.newPage();
  await page.goto("https://portalcfdi.facturaelectronica.sat.gob.mx/");
  await page.waitForNetworkIdle();
  setInterval(async () => {
    await page.goto("https://portalcfdi.facturaelectronica.sat.gob.mx/");
    await page.waitForNetworkIdle();
    await saveImage(page);
  }, 20000);
  //   await page.screenshot({ captureBeyondViewport: true, path: "hn.png" });
  //   await browser.close();
};
async function saveImage(page) {
  const imagedata = await page.$eval("#divCaptcha > img", (e) => e.src);
  // Grab the extension to resolve any image error
  var ext = imagedata.split(";")[0].match(/jpeg|png|gif/)[0];
  // strip off the data: url prefix to get just the base64-encoded bytes
  var data = imagedata.replace(/^data:image\/\w+;base64,/, "");
  var buf = new Buffer(data, "base64");
  writeFile(`images/${randomUUID()}.${ext}`, buf, (err) => {
    console.log(err);
  });
}
getImage();
