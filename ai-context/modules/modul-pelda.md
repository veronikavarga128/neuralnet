<!--
================================================================
  AI-NAK SZÁNT FÁJL — KITÖLTÖTT PÉLDA-MODUL
  Ez egy PÉLDA arra, hogyan néz ki egy kész modul-doc (a
  _MODULE_TEMPLATE.md kitöltve). Éles projektben töröld, vagy írd
  át a valódi moduljaidra.
================================================================
-->

# Auth (példa modul)

## Cél

A felhasználók bejelentkezését, session-kezelését és jogosultság-ellenőrzését
végzi. Minden védett endpoint ezen keresztül dönt hozzáférésről.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `auth/models.py` | User modell és szerepkörök |
| `auth/views.py` | Login / logout / profil nézetek |
| `auth/middleware.py` | Session-ellenőrzés minden kérésnél |

## Belépési pontok

- `login(request)` — belépési nézet
- `@require_role("admin")` — dekorátor jogosultság-ellenőrzéshez

## Függőségek

Támaszkodik: a projekt session-store-jára és a DB `users` táblájára.
Rá támaszkodik: gyakorlatilag minden védett modul (contracts, cases, ...).

## Buktatók

- A session-cookie neve környezetenként eltér — mindig az env-változóból olvasd.
- Új szerepkör felvételekor a migrációt NE felejtsd el.

## Hogyan teszteld

```
pytest auth/tests/ -v
```

Kézi: jelentkezz be teszt-userrel, majd nyiss meg egy admin-only oldalt
nem-admin userrel — 403-at kell kapnod.
