const CACHE_NAME = 'AcademicDissect-v1';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Opened cache');
      return cache.addAll([
        '/CVPR2025/',
        '/CVPR2025/index.html',
        '/CVPR2025/favicon.ico',
        '/CVPR2025/cvpr.json',
        '/CVPR2025/searchWorker.js'
      ]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }
  event.respondWith(
    caches.match(event.request, { ignoreSearch: true })
      .then((response) => response || fetch(event.request).then((fetchResponse) => {
        if (!fetchResponse || fetchResponse.status !== 200 || fetchResponse.type !== 'basic') {
          return fetchResponse;
        }
        const responseToCache = fetchResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });
        return fetchResponse;
      }))
  );
});

self.addEventListener('activate', (event) => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames.map((cacheName) => {
          if (!cacheWhitelist.includes(cacheName)) {
            return caches.delete(cacheName);
          }
        })
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
