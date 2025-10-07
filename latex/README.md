# Proyecto LaTeX con Tablas Importadas

Este proyecto demuestra cómo crear un documento LaTeX que importa tablas desde archivos `.tex` externos.

## Archivos del Proyecto

- `main.tex`: Documento principal LaTeX
- `tabla.tex`: Archivo con las tablas a importar
- `README.md`: Este archivo de documentación

## Estructura del Proyecto

```
latex/
├── main.tex       # Documento principal
├── tabla.tex      # Tablas externas
└── README.md      # Documentación
```

## Cómo Funciona

El documento principal (`main.tex`) usa el comando `\input{tabla.tex}` para importar las tablas definidas en el archivo externo. Esto permite:

- **Modularidad**: Separar el contenido de las tablas del documento principal
- **Reutilización**: Usar las mismas tablas en múltiples documentos
- **Mantenimiento**: Actualizar tablas sin tocar el documento principal
- **Organización**: Mantener archivos más pequeños y manejables

## Compilación

Para compilar el documento, ejecuta:

```bash
pdflatex main.tex
```

Si necesitas bibliografía o referencias cruzadas, ejecuta:

```bash
pdflatex main.tex
pdflatex main.tex
```

## Personalización

### Modificar las Tablas

Edita el archivo `tabla.tex` para cambiar el contenido de las tablas. Puedes:

- Cambiar los datos
- Añadir más filas o columnas
- Modificar el formato y estilo
- Añadir más tablas

### Crear Nuevas Tablas

Para crear tablas adicionales:

1. Crea un nuevo archivo `.tex` (ej: `tabla2.tex`)
2. Define tus tablas en ese archivo
3. Impórtalas en `main.tex` usando `\input{tabla2.tex}`

### Ejemplo de Nueva Tabla

Crea `tabla2.tex`:

```latex
\begin{table}[H]
    \centering
    \caption{Mi Nueva Tabla}
    \label{tab:nueva}
    \begin{tabular}{@{}lcc@{}}
        \toprule
        \textbf{Item} & \textbf{Valor A} & \textbf{Valor B} \\
        \midrule
        Elemento 1 & 100 & 200 \\
        Elemento 2 & 150 & 250 \\
        \bottomrule
    \end{tabular}
\end{table}
```

Y en `main.tex` añade:
```latex
\input{tabla2.tex}
```

## Paquetes Utilizados

- `booktabs`: Para tablas profesionales
- `array`: Funcionalidades adicionales para tablas
- `multirow`, `multicol`: Celdas que abarcan múltiples filas/columnas
- `xcolor`: Colores para tablas
- `float`: Control de posicionamiento
- `caption`: Personalización de títulos

## Consejos

1. **Posicionamiento**: Usa `[H]` para posición fija o `[htbp]` para flotante
2. **Referencias**: Usa `\label{}` y `\ref{}` para referenciar tablas
3. **Títulos**: Siempre incluye `\caption{}` para describir la tabla
4. **Formato**: Usa `\toprule`, `\midrule`, `\bottomrule` para líneas profesionales

## Solución de Problemas

- **Error "File not found"**: Asegúrate de que `tabla.tex` esté en el mismo directorio
- **Tabla fuera de página**: Usa `longtable` para tablas largas
- **Formato incorrecto**: Verifica el número de columnas en `\begin{tabular}{}`