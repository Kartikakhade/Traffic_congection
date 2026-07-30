const fs = require("fs");
const zlib = require("zlib");

const pdfPath = process.argv[2];

if (!pdfPath) {
  console.error("Usage: node tools/pdf_extract_snippets.js <pdf-path>");
  process.exit(1);
}

const data = fs.readFileSync(pdfPath);
const text = data.toString("latin1");
const streamRegex = /stream\r?\n/g;

let match;
let index = 0;

while ((match = streamRegex.exec(text)) !== null) {
  const streamStart = match.index + match[0].length;
  const endIdx = text.indexOf("endstream", streamStart);
  if (endIdx === -1) {
    continue;
  }

  const raw = data.subarray(streamStart, endIdx);
  let decoded;
  let encoding = "raw";

  try {
    decoded = zlib.inflateSync(raw);
    encoding = "flate";
  } catch {
    decoded = raw;
  }

  const decodedText = decoded.toString("latin1");
  const interesting =
    /(?:\bBT\b|\bET\b|\bTj\b|\bTJ\b|Accuracy|accuracy|dataset|CNN|LSTM|hybrid|abstract|conclusion|proposed|results)/i.test(
      decodedText,
    );

  if (interesting) {
    const snippet = decodedText
      .replace(/[^\x09\x0A\x0D\x20-\x7E]/g, " ")
      .replace(/\s+/g, " ")
      .slice(0, 2000);

    console.log(`--- stream ${index} (${encoding}) ---`);
    console.log(snippet);
    console.log();
  }

  index += 1;
}
