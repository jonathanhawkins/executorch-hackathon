/** @type {import('next').NextConfig} */
const nextConfig = {
  // Nothing here on purpose. "output: standalone" conflicts with `next start`
  // and is only useful for container images, which this is not.
};

export default nextConfig;
