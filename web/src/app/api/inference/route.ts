import { spawn } from "node:child_process";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function runPythonInference(
  pythonExecutable: string,
  pythonScript: string,
  projectRoot: string,
  payload: unknown,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable, [pythonScript], {
      cwd: projectRoot,
      env: { ...process.env, AML_PROJECT_ROOT: projectRoot },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Python inference melebihi batas waktu 120 detik."));
    }, 120_000);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => { stdout += chunk; });
    child.stderr.on("data", (chunk: string) => { stderr += chunk; });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(stderr || stdout || `Python berhenti dengan exit code ${code}.`));
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

/**
 * This route deliberately delegates scoring to Python.  The saved artifact is
 * a scikit-learn/Joblib Local Outlier Factor bundle, so recreating its
 * preprocessing or scoring in TypeScript would not be the trained model.
 */
export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const webRoot = process.cwd();
    const projectRoot = path.resolve(webRoot, "..");
    const pythonScript = path.join(webRoot, "python", "inference_service.py");
    const pythonExecutable =
      process.env.AML_PYTHON_EXECUTABLE ?? "E:\\Anaconda3\\envs\\super\\python.exe";

    const stdout = await runPythonInference(
      pythonExecutable,
      pythonScript,
      projectRoot,
      payload,
    );

    const result = JSON.parse(stdout) as { ok: boolean; data?: unknown; error?: string };
    if (!result.ok) {
      return Response.json({ error: result.error ?? "Inference tidak berhasil." }, { status: 400 });
    }
    return Response.json(result.data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Inference tidak berhasil.";
    return Response.json(
      {
        error: `Tidak dapat memuat atau menjalankan model tersimpan: ${message}`,
      },
      { status: 500 },
    );
  }
}
