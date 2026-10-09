# Bundled engine: Stockfish 18 for Windows

`stockfish-18-windows.zip` contains `stockfish.exe`, a Windows x86-64 build of
[Stockfish](https://stockfishchess.org/) 18. ChessCore unpacks it into
`engines/stockfish/` the first time the `match` mode needs it (that folder is
not committed). Setting `CHESSCORE_ENGINE` to another engine overrides it.

The executable is zipped because it is larger than GitHub's 100 MB file limit.

## Build

Built from the unmodified official source, tag
[`sf_18`](https://github.com/official-stockfish/Stockfish/tree/sf_18)
(commit cb3d4ee9), with the default networks `nn-c288c895ea92.nnue` and
`nn-37f18f62d772.nnue` embedded:

```
make build ARCH=x86-64-sse41-popcnt COMP=mingw
```

`x86-64-sse41-popcnt` runs on practically every 64-bit PC from the last
fifteen years, including ones without AVX2. It is slower than the AVX2 build,
which does not matter here because ChessCore limits Stockfish's thinking time.

SHA-256 of `stockfish.exe`:
`c7465ddd1a09b97f69daeef2bb58919af4557e4fc332a63a2c044088fd17b82b`

## Licence

Stockfish is free software licensed under the GNU General Public License
version 3. The licence text is in `Copying.txt` (also inside the zip, together
with the list of authors). The source code is available at
https://github.com/official-stockfish/Stockfish.
