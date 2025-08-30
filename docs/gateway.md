# Gateway & Traffic Management

Options:
- **NGINX Ingress**: simple, popular; great default.
- **Istio/Envoy**: advanced routing, mTLS, circuit breaking; good for canaries & mesh observability.
- **Kong/APISIX**: plugin ecosystems, auth/rate-limit; managed options available.

Start simple (NGINX), add mesh/gateway features as needs grow.
