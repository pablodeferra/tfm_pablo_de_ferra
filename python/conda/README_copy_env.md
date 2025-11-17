
# Instalador autocontenido para el entorno `cmb_py`

En esta carpeta ahora solo queda un instalador `.sh` y este `README`. Copia el archivo
`create_env_selfcontained.sh` al otro ordenador y ejecútalo para crear el entorno `cmb_py`.

Uso rápido

1. En el equipo destino, dar permisos y ejecutar:

```bash
chmod +x create_env_selfcontained.sh
./create_env_selfcontained.sh
```

2. Para usar otro nombre de entorno:

```bash
./create_env_selfcontained.sh --name mi_entorno
```

Requisitos y notas

- Debe existir `conda` en el sistema destino (Miniconda/Anaconda). El script comprobará y fallará si no lo encuentra.
- El instalador incluye tanto dependencias conda como pip. El script elimina la línea `prefix:` para que el YAML embebido sea portátil.
- Si quieres una réplica binaria exacta (útil para equipos sin acceso a Internet), considera usar `conda-pack`; puedo añadir ese flujo si lo deseas.

Si quieres que deje algún script adicional en esta carpeta (por ejemplo para empaquetar con `conda-pack`), dímelo y lo añado. Si no, ya está listo: copia solo `create_env_selfcontained.sh` y este `README` cuando quieras transferir el entorno.
