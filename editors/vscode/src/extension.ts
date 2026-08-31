/**
 * Offset's editor companion.
 *
 * offset keeps its terminal identity; this surfaces three things VS Code is
 * genuinely better at than a TUI — a persistent status line, a real side-by-side
 * diff with per-file accept, and sending a selection with its file context.
 * It deliberately does not reimplement the agent UI: there is one conversation
 * and it lives in the terminal.
 *
 * The bridge is authenticated and can apply edits, so every failure path here
 * ends in a status-bar state and a message, never a silent retry loop. offset
 * not running is the normal case, not an error.
 */

import * as path from "path";
import * as vscode from "vscode";

import { BridgeClient, ChangePayload, offsetHome, StatusPayload } from "./bridge";

/** A row in the Offset view: either a heading or a leaf with a command. */
class Row extends vscode.TreeItem {
  constructor(
    label: string,
    description?: string,
    icon?: string,
    command?: vscode.Command,
    collapsible = vscode.TreeItemCollapsibleState.None,
  ) {
    super(label, collapsible);
    this.description = description;
    if (icon) {
      this.iconPath = new vscode.ThemeIcon(icon);
    }
    this.command = command;
  }
}

class ActivityView implements vscode.TreeDataProvider<Row> {
  private readonly changed = new vscode.EventEmitter<Row | undefined>();
  public readonly onDidChangeTreeData = this.changed.event;

  public status?: StatusPayload;
  public changes: ChangePayload[] = [];
  public problem?: string;

  public refresh(): void {
    this.changed.fire(undefined);
  }

  public getTreeItem(element: Row): vscode.TreeItem {
    return element;
  }

  public getChildren(): Row[] {
    if (this.problem) {
      return [
        new Row(this.problem, undefined, "warning", {
          command: "offset.connect",
          title: "Connect",
        }),
      ];
    }
    const rows: Row[] = [];
    const status = this.status;
    if (status) {
      rows.push(new Row("model", status.model ?? "unknown", "hubot"));
      rows.push(new Row("state", status.state ?? "idle", "pulse"));
      if (status.session) {
        rows.push(
          new Row("session", path.basename(status.session), "history", {
            command: "offset.openSession",
            title: "Open session",
          }),
        );
      }
    }
    if (this.changes.length) {
      rows.push(
        new Row(
          `${this.changes.length} pending change(s)`,
          undefined,
          "diff",
          { command: "offset.showDiff", title: "Show changes" },
        ),
      );
      for (const change of this.changes) {
        const counts =
          change.additions !== undefined || change.deletions !== undefined
            ? `+${change.additions ?? 0} -${change.deletions ?? 0}`
            : change.status;
        rows.push(new Row(`  ${change.path}`, counts, "file"));
      }
    }
    for (const job of status?.jobs ?? []) {
      rows.push(new Row(`job ${job.id.slice(0, 8)}`, job.state, "server-process"));
    }
    if (!rows.length) {
      rows.push(new Row("connected; nothing running", undefined, "check"));
    }
    return rows;
  }
}

/** Renders the agent's pending changes so a diff can be opened read-only. */
class DiffProvider implements vscode.TextDocumentContentProvider {
  public static readonly scheme = "offset-original";
  private readonly bodies: Record<string, string> = {};

  public set(target: string, text: string): vscode.Uri {
    this.bodies[target] = text;
    return vscode.Uri.parse(`${DiffProvider.scheme}:${target}`);
  }

  public provideTextDocumentContent(uri: vscode.Uri): string {
    return this.bodies[uri.path] ?? "";
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const client = new BridgeClient();
  const view = new ActivityView();
  const diffs = new DiffProvider();
  const bar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  const log = vscode.window.createOutputChannel("Offset");
  let retry: NodeJS.Timeout | undefined;

  bar.command = "offset.connect";
  bar.show();

  const paint = () => {
    const icons: Record<string, string> = {
      offline: "$(circle-slash)",
      connecting: "$(sync~spin)",
      ready: "$(check)",
    };
    const icon = icons[client.state] ?? "$(question)";
    const model = view.status?.model ? ` ${view.status.model}` : "";
    const jobs = view.status?.jobs?.length ?? 0;
    const badge = jobs ? ` $(server-process)${jobs}` : "";
    bar.text = `${icon} offset${model}${badge}`;
    bar.tooltip =
      client.state === "ready"
        ? `offset ${view.status?.state ?? "idle"} — click to refresh`
        : view.problem ?? "offset is not running";
  };

  const sync = async () => {
    if (client.state !== "ready") {
      return;
    }
    try {
      view.status = await client.status();
      view.changes = await client.diff();
      view.problem = undefined;
    } catch (err) {
      view.problem = (err as Error).message;
    }
    paint();
    view.refresh();
  };

  const settings = () => vscode.workspace.getConfiguration("offset");

  const connect = async (announce: boolean) => {
    if (retry) {
      clearTimeout(retry);
      retry = undefined;
    }
    try {
      await client.connect(offsetHome(settings().get<string>("home")));
      view.problem = undefined;
      log.appendLine(`connected to offset (pid ${client.descriptor?.pid ?? "?"})`);
      await sync();
      if (announce) {
        vscode.window.setStatusBarMessage("$(check) offset connected", 2000);
      }
    } catch (err) {
      const message = (err as Error).message;
      view.problem = message;
      view.status = undefined;
      view.changes = [];
      log.appendLine(`not connected: ${message}`);
      paint();
      view.refresh();
      if (announce) {
        vscode.window.showWarningMessage(
          `Offset: ${message}`,
          "Retry",
        ).then((choice) => {
          if (choice === "Retry") {
            void connect(true);
          }
        });
      }
    }
  };

  client.on("state", paint);
  client.on("closed", (why: string) => {
    log.appendLine(`connection lost: ${why}`);
    view.problem = why;
    view.status = undefined;
    view.changes = [];
    paint();
    view.refresh();
    const seconds = settings().get<number>("reconnectSeconds") ?? 5;
    if (seconds > 0 && !retry) {
      retry = setTimeout(() => {
        retry = undefined;
        void connect(false);
      }, seconds * 1000);
    }
  });
  client.on("event", (method: string) => {
    log.appendLine(`event ${method}`);
    void sync();
  });

  context.subscriptions.push(
    bar,
    log,
    vscode.window.registerTreeDataProvider("offsetActivity", view),
    vscode.workspace.registerTextDocumentContentProvider(DiffProvider.scheme, diffs),

    vscode.commands.registerCommand("offset.connect", () => connect(true)),
    vscode.commands.registerCommand("offset.refresh", () => sync()),

    vscode.commands.registerCommand("offset.cancel", async () => {
      try {
        await client.request("cancel");
        vscode.window.setStatusBarMessage("$(debug-stop) offset: cancelled", 2000);
      } catch (err) {
        vscode.window.showErrorMessage(`Offset: ${(err as Error).message}`);
      }
    }),

    vscode.commands.registerCommand("offset.prompt", async () => {
      const editor = vscode.window.activeTextEditor;
      const selection = editor?.document.getText(editor.selection) ?? "";
      const asked = await vscode.window.showInputBox({
        prompt: selection ? "What should offset do with the selection?" : "Send to offset",
        placeHolder: "explain this, or add a test for it",
      });
      if (!asked) {
        return;
      }
      // The file and line are worth sending: the agent is in a terminal and
      // cannot see which buffer is focused.
      let text = asked;
      if (editor && selection) {
        const where = vscode.workspace.asRelativePath(editor.document.uri);
        const line = editor.selection.start.line + 1;
        text = `${asked}\n\n${where}:${line}\n\`\`\`\n${selection}\n\`\`\``;
      }
      try {
        await client.request("prompt", { text });
        vscode.window.setStatusBarMessage("$(comment) sent to offset", 2000);
      } catch (err) {
        vscode.window.showErrorMessage(`Offset: ${(err as Error).message}`);
      }
    }),

    vscode.commands.registerCommand("offset.showDiff", async () => {
      await sync();
      if (!view.changes.length) {
        vscode.window.showInformationMessage("Offset: no pending changes");
        return;
      }
      const picked = await vscode.window.showQuickPick(
        view.changes.map((change) => ({
          label: change.path,
          description: `${change.status} +${change.additions ?? 0} -${change.deletions ?? 0}`,
          change,
        })),
        { placeHolder: "Which change?" },
      );
      if (!picked) {
        return;
      }
      const folders = vscode.workspace.workspaceFolders;
      if (!folders?.length) {
        vscode.window.showWarningMessage("Offset: open a folder to diff against");
        return;
      }
      const target = vscode.Uri.joinPath(folders[0].uri, picked.change.path);
      // The bridge reports the diff, not the original body, so the left-hand
      // side is the diff itself: honest about what is known rather than
      // reconstructing a file we were never sent.
      const left = diffs.set(
        `/${picked.change.path}.diff`,
        picked.change.diff ?? "(no diff text was reported)",
      );
      await vscode.commands.executeCommand(
        "vscode.diff",
        left,
        target,
        `offset: ${picked.change.path}`,
      );
    }),

    vscode.commands.registerCommand("offset.openSession", async () => {
      try {
        const sessions = await client.sessions();
        if (!sessions.length) {
          vscode.window.showInformationMessage("Offset: no sessions yet");
          return;
        }
        const picked = await vscode.window.showQuickPick(
          sessions.map((entry) => ({
            label: String(entry.id ?? "?"),
            description: `${entry.messages ?? 0} messages`,
            detail: String(entry.first_line ?? ""),
            entry,
          })),
          { placeHolder: "Which session?" },
        );
        const target = picked?.entry.path;
        if (typeof target === "string") {
          const document = await vscode.workspace.openTextDocument(vscode.Uri.file(target));
          await vscode.window.showTextDocument(document);
        }
      } catch (err) {
        vscode.window.showErrorMessage(`Offset: ${(err as Error).message}`);
      }
    }),

    { dispose: () => {
        clearTimeout(retry);
        client.dispose();
      } },
  );

  paint();
  if (settings().get<boolean>("autoConnect") !== false) {
    void connect(false);
  }
}

export function deactivate(): void {
  // Every disposable, including the bridge socket, is registered on the
  // context, so there is nothing left to do here.
}
