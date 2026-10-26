#!/bin/sh
# Optional read-only token Secret. Application logs never print this value.
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "${NRP_GIT_TOKEN:-}" ;;
  *) exit 1 ;;
esac
