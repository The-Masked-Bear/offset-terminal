/**
 * Finding a running offset.
 *
 * Three steps, in the order the Python side documents them: resolve the home
 * directory, read the descriptor it publishes, then read the token from the
 * path the descriptor names. The token is never inlined in the descriptor -
 * descriptors get pasted into bug reports.
 *
 * Nothing here throws. "offset is not running" is the ordinary case, not an
 * error worth a stack trace, so every failure comes back as a sentence a
 * status bar can display.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { BRIDGE_VERSION, Connection, DESCRIPTOR_NAME, Descriptor } from "./protocol.js";

/**
 * Where offset keeps its state, honouring the same precedence the CLI does.
 *
 * `$OFFSET_HOME` wins so that a test, a `--home` run and a second checkout can
 * each have their own; otherwise `~/.offset`.
 */
export function offsetHome(configured?: string): string {
  const chosen = configured?.trim();
  if (chosen) return path.resolve(chosen);
  const fromEnv = process.env.OFFSET_HOME?.trim();
  if (fromEnv) return path.resolve(fromEnv);
  return path.join(os.homedir(), ".offset");
}

export interface Discovery {
  connection?: Connection;
  /** Why no connection could be described, in words a user can act on. */
  problem?: string;
}

/**
 * Read the descriptor and its token.
 *
 * A descriptor whose version this client does not know is refused rather than
 * guessed at: connecting with the wrong assumptions produces a confusing
 * failure much later, and "upgrade one of the two halves" is actionable now.
 */
export function discover(home?: string): Discovery {
  const root = offsetHome(home);
  const descriptorPath = path.join(root, DESCRIPTOR_NAME);

  let raw: string;
  try {
    raw = fs.readFileSync(descriptorPath, "utf8");
  } catch {
    return { problem: `no bridge descriptor at ${descriptorPath}; is offset running?` };
  }

  let parsed: Descriptor;
  try {
    parsed = JSON.parse(raw) as Descriptor;
  } catch {
    return { problem: `${descriptorPath} is not valid JSON` };
  }
  if (!parsed || typeof parsed !== "object") {
    return { problem: `${descriptorPath} does not contain a JSON object` };
  }

  if (parsed.version && parsed.version !== BRIDGE_VERSION) {
    return {
      problem:
        `this client speaks bridge version ${BRIDGE_VERSION} and offset published ` +
        `${parsed.version}; upgrade whichever is older`,
    };
  }

  const tokenPath = parsed.token_path;
  if (!tokenPath) {
    return { problem: `${descriptorPath} names no token path` };
  }

  let token: string;
  try {
    token = fs.readFileSync(tokenPath, "utf8").trim();
  } catch {
    return { problem: `the bridge token at ${tokenPath} could not be read` };
  }
  if (!token) {
    return { problem: `the bridge token at ${tokenPath} is empty` };
  }

  const problem = addressProblem(parsed);
  if (problem) return { problem };

  return { connection: { ...parsed, token } };
}

/** Whether the descriptor actually says where to connect. */
function addressProblem(descriptor: Descriptor): string | undefined {
  if (descriptor.transport === "unix") {
    // The field is `path`. Reading `socket` here is a real bug that has been
    // written before: it silently falls through to the TCP branch and connects
    // to port 0, whose failure carries no message at all.
    return descriptor.path ? undefined : "the descriptor claims a unix socket but names no path";
  }
  if (descriptor.transport === "tcp") {
    return descriptor.port ? undefined : "the descriptor claims tcp but names no port";
  }
  return `unknown transport ${String(descriptor.transport)}`;
}

/** A one-line description of where a connection points, for logs and status. */
export function describe(connection: Descriptor): string {
  return connection.transport === "unix"
    ? String(connection.path)
    : `${connection.host || "127.0.0.1"}:${connection.port}`;
}
