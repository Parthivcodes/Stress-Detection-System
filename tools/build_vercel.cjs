// Publish only website assets. Never expose the repository root as static output.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'public');
// Refuse an existing output instead of deleting possibly user-owned files.
if (fs.existsSync(output)) throw new Error('public/ already exists; use a clean build checkout.');
fs.mkdirSync(output);
const allowed = ['index.html','workspace.html','landing.css','landing.js',
  'prediction.css','prediction.js','analysis-pdf.js','three.min.js','fonts'];
for (const name of allowed) fs.cpSync(path.join(root,'frontend',name),path.join(output,name),{recursive:true});
console.log('Packaged public frontend only; no data, reports or models included.');
