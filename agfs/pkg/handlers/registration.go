package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"net/http"

	"github.com/c4pt0r/agfs/agfs-server/pkg/filesystem"
	"github.com/c4pt0r/agfs/agfs-server/pkg/mountablefs"
)

const registrationHeader = "X-AGFS-Registration"

// On disconnect, stop admitting tagged operations and drain those already
// admitted before releasing the cross-process storage lock.
type registrationSession struct {
	lock    *mountablefs.Registration
	handler http.Handler
}

func (h *Handler) registration(w http.ResponseWriter, r *http.Request) {
	locker, ok := h.fs.(*mountablefs.MountableFS)
	if !ok {
		writeError(w, http.StatusNotImplemented, "registration sessions unavailable")
		return
	}
	locked, err := locker.LockRegistration(r.URL.Query().Get("path"))
	if err != nil {
		switch {
		case errors.Is(err, filesystem.ErrRegistrationBusy):
			writeError(w, http.StatusConflict, err.Error())
		case errors.Is(err, mountablefs.ErrRegistrationUnsupported):
			writeError(w, http.StatusNotImplemented, err.Error())
		default:
			writeError(w, http.StatusInternalServerError, "cannot acquire registration session")
		}
		return
	}
	defer locked.Close()
	var id [32]byte
	if _, err := rand.Read(id[:]); err != nil {
		writeError(w, http.StatusInternalServerError, "cannot initialize registration session")
		return
	}
	token := hex.EncodeToString(id[:])
	session := &registrationSession{lock: locked}
	mux := http.NewServeMux()
	NewHandler(locked.FS, h.trafficMonitor).setupFileRoutes(mux, "/api/v1/registration-session")
	session.handler = mux
	h.registrations.Store(token, session)
	defer h.registrations.Delete(token)
	w.Header().Set(registrationHeader, token)
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	if http.NewResponseController(w).Flush() == nil {
		<-r.Context().Done()
	}
}

func (h *Handler) registrationOperation(w http.ResponseWriter, r *http.Request) {
	value, ok := h.registrations.Load(r.Header.Get(registrationHeader))
	if !ok {
		writeError(w, http.StatusConflict, "registration session closed")
		return
	}
	session := value.(*registrationSession)
	done, ok := session.lock.Enter()
	if !ok {
		writeError(w, http.StatusConflict, "registration session closed")
		return
	}
	defer done()
	session.handler.ServeHTTP(w, r)
}
