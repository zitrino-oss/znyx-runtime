#!/usr/bin/env node
/**
 * znyx.js — shim that executes the downloaded platform binary.
 *
 * If the binary is missing (unsupported platform or failed download) it prints
 * clear guidance and exits non-zero so the user knows what to do.
 */
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ext = process.platform === 'win32' ? '.exe' : '';
const binaryPath = path.join(__dirname, `znyx-bin${ext}`);

if (!fs.existsSync(binaryPath)) {
  console.error(
    '\n[znyx] The ZNYX Runtime binary is not available on this platform.\n\n' +
    '       Install options:\n' +
    '         pip install znyx-runtime    (Python 3.9+)\n' +
    '         docker run znyx/runtime\n'
  );
  process.exit(1);
}

const result = spawnSync(binaryPath, process.argv.slice(2), { stdio: 'inherit' });

if (result.error) {
  console.error(`[znyx] Failed to execute binary: ${result.error.message}`);
  process.exit(1);
}

process.exit(result.status ?? 0);
