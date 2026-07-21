import Link from "next/link";

export function AppHeader() {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link className="brand" href="/" aria-label="AML Detection Studio home">
          <span className="brand-mark">AML</span>
          <span>
            <strong>DETECTION STUDIO</strong>
            <small>RULES · ANOMALY · EXPLAINABILITY</small>
          </span>
        </Link>
        <nav className="primary-nav" aria-label="Navigasi utama">
          <Link href="/evaluation">Page 1 · Evaluasi</Link>
          <Link href="/inference">Page 2 · Inference</Link>
        </nav>
      </div>
    </header>
  );
}
