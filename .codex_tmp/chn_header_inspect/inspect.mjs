import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath =
  "D:/Code/Python/GeneralParameter/J35_001_none/CONFIG/ChannelDetail/ChnTableHeader.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 20000,
  tableMaxRows: 120,
  tableMaxCols: 30,
  tableMaxCellChars: 120,
});
process.stdout.write(`${summary.ndjson}\n`);

const firstSheet = workbook.worksheets.getItemAt(0);
const preview = await workbook.render({
  sheetName: firstSheet.name,
  autoCrop: "all",
  scale: 1.2,
  format: "png",
});
await fs.writeFile(
  "D:/Code/Python/surrogate_model/.codex_tmp/chn_header_inspect/preview.png",
  new Uint8Array(await preview.arrayBuffer()),
);
