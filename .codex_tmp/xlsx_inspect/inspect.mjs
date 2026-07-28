import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "D:/Code/Python/surrogate_model/cli/极点频段配置.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const overview = await workbook.inspect({
  kind: "sheet,table",
  maxChars: 5000,
  tableMaxRows: 12,
  tableMaxCols: 10,
  tableMaxCellChars: 100,
});
process.stdout.write(`${overview.ndjson}\n`);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 2000,
});
process.stdout.write(`${sheets.ndjson}\n`);

const preview = await workbook.render({
  sheetName: "Sheet",
  autoCrop: "all",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  "D:/Code/Python/surrogate_model/.codex_tmp/xlsx_inspect/preview.png",
  new Uint8Array(await preview.arrayBuffer()),
);
