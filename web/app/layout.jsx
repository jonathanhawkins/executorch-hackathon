import "./globals.css";

export const metadata = {
  title: "VENTUNO Q · On-device vision",
  description: "InternVL3 running on the Hexagon NPU, with nothing leaving the board",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
