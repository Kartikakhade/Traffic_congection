const fs = require("fs");

const pdfPath = process.argv[2];

if (!pdfPath) {
  console.error("Usage: node tools/pdf_probe.js <pdf-path>");
  process.exit(1);
}

const data = fs.readFileSync(pdfPath);
const text = data.toString("latin1");

const count = (pattern) => {
  const matches = text.match(pattern);
  return matches ? matches.length : 0;
};

console.log(JSON.stringify({
  path: pdfPath,
  size: data.length,
  flateDecode: count(/\/FlateDecode/g),
  toUnicode: count(/\/ToUnicode/g),
  bt: count(/\bBT\b/g),
  tj: count(/\bTj\b/g),
  TJ: count(/\bTJ\b/g),
  obj: count(/\bendobj\b/g),
  stream: count(/\bstream\b/g),
}, null, 2));
