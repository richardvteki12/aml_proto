import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";

export default function Home() {
  return (
    <>
      <AppHeader />
      <main className="welcome-shell">
        <section className="welcome-card" aria-labelledby="welcome-title">
          <p className="eyebrow">AML DETECTION STUDIO</p>
          <h1 id="welcome-title">Selamat datang.</h1>
          <p className="lede">
            Gunakan dua halaman berikut untuk melihat hasil evaluasi data sintetik dan mencoba
            inference terhadap transaksi baru dengan rule AML serta model anomaly detection yang
            sudah terlatih.
          </p>
          <div className="welcome-actions">
            <Link className="button button-primary" href="/evaluation">
              Lihat evaluasi
            </Link>
            <Link className="button button-secondary" href="/inference">
              Coba inference
            </Link>
          </div>
          <p className="welcome-note">
            Model pada simulasi memuat artefak tersimpan. Tidak ada proses training ulang saat
            inference dijalankan.
          </p>
        </section>
      </main>
    </>
  );
}
