const fs = require("fs");
const zlib = require("zlib");

const pdfPath = process.argv[2];
const outPath = process.argv[3];

if (!pdfPath) {
  console.error("Usage: node tools/pdf_to_text.js <pdf-path> [out-path]");
  process.exit(1);
}

const data = fs.readFileSync(pdfPath);
const rawText = data.toString("latin1");
const streamRegex = /stream\r?\n/g;

const ligatureMap = {
  "\u0000": "-",
  "\u0001": "ff",
  "\u0002": "fi",
  "\u0003": "fl",
  "\u0004": "ffi",
  "\u0005": "ffl",
  "\u0016": "–",
  "\u0017": "—",
  "\u0093": "\"",
  "\u0094": "\"",
};

const decodePdfString = (value) => {
  let result = "";

  for (let i = 0; i < value.length; i += 1) {
    const ch = value[i];

    if (ch !== "\\") {
      result += ligatureMap[ch] ?? ch;
      continue;
    }

    const next = value[i + 1];

    if (next === undefined) {
      break;
    }

    if (/[0-7]/.test(next)) {
      let oct = next;
      let consumed = 1;

      while (consumed < 3 && /[0-7]/.test(value[i + 1 + consumed] ?? "")) {
        oct += value[i + 1 + consumed];
        consumed += 1;
      }

      const decoded = String.fromCharCode(parseInt(oct, 8));
      result += ligatureMap[decoded] ?? decoded;
      i += consumed;
      continue;
    }

    const escapedMap = {
      n: "\n",
      r: "\r",
      t: "\t",
      b: "\b",
      f: "\f",
      "(": "(",
      ")": ")",
      "\\": "\\",
    };

    result += escapedMap[next] ?? next;
    i += 1;
  }

  return result;
};

const extractFromArray = (arrayContent) => {
  const parts = [];
  const tokenRegex = /\((?:\\.|[^\\)])*\)|-?\d+(?:\.\d+)?/g;
  let match;

  while ((match = tokenRegex.exec(arrayContent)) !== null) {
    const token = match[0];

    if (token.startsWith("(")) {
      const inner = token.slice(1, -1);
      parts.push({ type: "text", value: decodePdfString(inner) });
      continue;
    }

    parts.push({ type: "num", value: Number(token) });
  }

  let line = "";

  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];

    if (part.type !== "text") {
      continue;
    }

    line += part.value;

    const next = parts[i + 1];
    if (!next) {
      continue;
    }

    if (next.type === "num" && Math.abs(next.value) >= 100) {
      line += " ";
    }
  }

  return line;
};

const extractFromStream = (streamText) => {
  const lines = [];
  const opRegex = /\[(.*?)\]\s*TJ|\((?:\\.|[^\\)])*\)\s*Tj/gms;
  let match;

  while ((match = opRegex.exec(streamText)) !== null) {
    if (match[1] !== undefined) {
      const line = extractFromArray(match[1]).trim();
      if (line) {
        lines.push(line);
      }
      continue;
    }

    const single = match[0];
    const start = single.indexOf("(");
    const end = single.lastIndexOf(")");
    const line = decodePdfString(single.slice(start + 1, end)).trim();
    if (line) {
      lines.push(line);
    }
  }

  return lines;
};

let match;
let pageLikeIndex = 0;
const blocks = [];

while ((match = streamRegex.exec(rawText)) !== null) {
  const start = match.index + match[0].length;
  const end = rawText.indexOf("endstream", start);
  if (end === -1) {
    continue;
  }

  const raw = data.subarray(start, end);
  let decoded;

  try {
    decoded = zlib.inflateSync(raw);
  } catch {
    decoded = raw;
  }

  const decodedText = decoded.toString("latin1");
  if (!decodedText.includes("BT")) {
    continue;
  }

  const lines = extractFromStream(decodedText);
  if (!lines.length) {
    continue;
  }

  pageLikeIndex += 1;
  blocks.push(`--- block ${pageLikeIndex} ---\n${lines.join("\n")}`);
}

const output = blocks.join("\n\n").replace(/[ \t]+\n/g, "\n");

if (outPath) {
  fs.writeFileSync(outPath, output, "utf8");
} else {
  process.stdout.write(output);
}
