package main

import "net/http"

// Tests can supply an ephemeral listener without changing the server's
// production configuration or exposing test control through the environment.
var serveHTTP = (*http.Server).ListenAndServe
