import * as SQLite from 'expo-sqlite';


const db = await SQLite.openDatabaseAsync('images');

await db.execAsync(`
  CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY NOT NULL,
    uri TEXT NOT NULL
  );
`);

for await (const row of db.getEachAsync('SELECT * FROM images')) {
  const typedRow = row as { id: number; value: string; intValue: number };
  console.log(typedRow.id, typedRow.value, typedRow.intValue);
}
