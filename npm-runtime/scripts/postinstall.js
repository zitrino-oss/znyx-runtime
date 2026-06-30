#!/usr/bin/env node
/**
 * postinstall.js — downloads the platform-matched ZNYX Runtime binary from
 * GitHub Releases, verifies its SHA256 checksum, and marks it executable.
 *
 * Supported platforms:
 *   macOS Apple Silicon  → znyx-runtime-darwin-arm64
 *   macOS Intel          → znyx-runtime-darwin-x64
 *   Linux x64            → znyx-runtime-linux-x64
 *   Linux ARM64          → znyx-runtime-linux-arm64
 *   Windows x64          → znyx-runtime-windows-x64.exe
 *
 * On unsupported platforms a clear message is printed directing the user to
 * `pip install znyx-runtime` or Docker. pip is never invoked automatically.
 */
'use strict';

const { execFileSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const https = require('https');
const os = require('os');
const path = require('path');

const PKG = require('../package.json');
const VERSION = PKG.version;
const GITHUB_REPO = 'zitrino/znyx';
const RELEASE_BASE = `https://github.com/${GITHUB_REPO}/releases/download/v${VERSION}`;

const PLATFORM_MAP = {
  'darwin-arm64': 'znyx-runtime-darwin-arm64',
  'darwin-x64':   'znyx-runtime-darwin-x64',
  'linux-x64':    'znyx-runtime-linux-x64',
  'linux-arm64':  'znyx-runtime-linux-arm64',
  'win32-x64':    'znyx-runtime-windows-x64.exe',
};

function getPlatformKey() {
  return `${os.platform()}-${os.arch()}`;
}

function getBinaryName() {
  return PLATFORM_MAP[getPlatformKey()] || null;
}

function getBinaryPath() {
  // Preserve .exe on Windows so the file is directly executable by the OS.
  const ext = os.platform() === 'win32' ? '.exe' : '';
  return path.join(__dirname, '..', 'bin', `znyx-bin${ext}`);
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    function fetch(u) {
      https.get(u, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          return fetch(res.headers.location);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`HTTP ${res.statusCode} downloading ${u}`));
        }
        res.pipe(file);
        file.on('finish', () => file.close(resolve));
      }).on('error', reject);
    }
    fetch(url);
  });
}

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  hash.update(fs.readFileSync(filePath));
  return hash.digest('hex');
}

async function main() {
  const binaryName = getBinaryName();

  if (!binaryName) {
    console.log(
      `\n[znyx] Unsupported platform: ${getPlatformKey()}\n` +
      `       Binary install is not available for this platform.\n\n` +
      `       Alternative install options:\n` +
      `         pip install znyx-runtime\n` +
      `         docker run znyx/runtime\n`
    );
    process.exit(0); // non-fatal — the package still installs
  }

  const binaryUrl = `${RELEASE_BASE}/${binaryName}`;
  const checksumUrl = `${RELEASE_BASE}/${binaryName}.sha256`;
  const binaryDest = getBinaryPath();
  const checksumDest = binaryDest + '.sha256';

  console.log(`[znyx] Downloading ZNYX Runtime v${VERSION} for ${getPlatformKey()}...`);

  try {
    await download(binaryUrl, binaryDest);
    await download(checksumUrl, checksumDest);

    const expected = fs.readFileSync(checksumDest, 'utf8').trim().split(/\s+/)[0];
    const actual = sha256File(binaryDest);

    if (actual !== expected) {
      fs.unlinkSync(binaryDest);
      fs.unlinkSync(checksumDest);
      throw new Error(
        `SHA256 mismatch!\n  expected: ${expected}\n  actual:   ${actual}\n` +
        `Binary removed. Re-run npm install to retry.`
      );
    }

    fs.chmodSync(binaryDest, 0o755);
    fs.unlinkSync(checksumDest);
    console.log(`[znyx] Binary installed and verified ✓`);
  } catch (err) {
    console.error(`\n[znyx] Could not download binary: ${err.message}`);
    console.error(
      `       Fallback: pip install znyx-runtime\n` +
      `                 docker run znyx/runtime\n`
    );
    // Exit 0 so npm install succeeds; the shim will print guidance at runtime.
    process.exit(0);
  }
}

main();
