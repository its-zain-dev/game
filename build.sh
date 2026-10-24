#!/bin/sh
set -e
cd ~/game
git add . && (git commit -m "update" || true) && git push
echo "ينتظر GitHub يبدأ..." && sleep 8
ID=$(gh run list -L1 --json databaseId -q '.[0].databaseId')
gh run watch "$ID" --exit-status
rm -rf pkg && gh run download "$ID" -n pkg -D pkg
ls -la pkg
sftp -P 2022 9aac48ba-3611-4014-95cc-469e5f7fa7cd.jtdu8tlq@fi17.bot-hosting.cloud <<'S'
-mkdir pkg
put pkg/client.js pkg/
put pkg/client_bg.wasm pkg/
put Cargo.toml
put src/client.rs src/
put src/main.rs src/
S
echo "✅ خلص. أعد تشغيل السيرفر من اللوحة"
