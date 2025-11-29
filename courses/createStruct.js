const fs = require("fs");
const path = require("path");

const ROOT_DIR = process.cwd(); // Run this from your project root
const OUTPUT_FILE = path.join(ROOT_DIR, "structure.json");

// Ignore these files/folders
const IGNORE = new Set(["node_modules", ".git", "structure.json"]);

function readDirRecursive(dir, basePath = "") {
  const result = {};

  const items = fs.readdirSync(dir, { withFileTypes: true });

  for (const item of items) {
    if (IGNORE.has(item.name)) continue;

    const fullPath = path.join(dir, item.name);
    const relativePath = path.join(basePath, item.name).replace(/\\/g, "/");

    if (item.isDirectory()) {
      const children = readDirRecursive(fullPath, relativePath);
      if (Object.keys(children).length > 0) {
        result[item.name] = children;
      }
    } 
    else if (item.isFile() && item.name.endsWith(".md")) {
      result[item.name] = relativePath;
    }
  }

  return result;
}

// Generate structure
const structure = readDirRecursive(ROOT_DIR);

// Write to JSON
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(structure, null, 2));

console.log("✅ Folder structure successfully generated into structure.json");
