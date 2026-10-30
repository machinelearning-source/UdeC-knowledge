# UdeC Knowledge Hub

Asistente RAG público para consultar los documentos oficiales de la universidad
(reglamentos, acuerdos, calendario académico y actas).

- Aplicación: https://udec-knowledge.streamlit.app/
- Catálogo público: https://machinelearning-source.github.io/UdeC-knowledge/
- Repositorio: https://github.com/machinelearning-source/UdeC-knowledge

## Cómo funciona

1. `core/ingestion.py` lee los PDFs de `data/documents/` y construye el índice
   vectorial FAISS en `vector_store/`.
2. `app.py` responde preguntas usando FastEmbed + FAISS + Groq, citando documento
   y página.
3. Cualquier persona puede subir o reemplazar un PDF desde la sección
   **Contribuir al conocimiento** de la aplicación.
4. El archivo se publica en GitHub y el workflow
   `.github/workflows/rebuild-index.yml` reconstruye `vector_store/`
   automáticamente.

## Activar el formulario público de contribución

El formulario necesita un token con permiso de escritura sobre el repositorio.
Sin ese token la aplicación sigue siendo pública, pero la sección de subida
muestra un aviso y no permite publicar.

1. Crea un token fine-grained en
   https://github.com/settings/personal-access-tokens/new
   - Repository access: **Only select repositories** → `UdeC-knowledge`
   - Permissions → Repository permissions → **Contents: Read and write**
   - Metadata: Read (se activa automáticamente)
2. En Streamlit Community Cloud abre la app → **Settings** → **Secrets** y pega:

   ```toml
   GITHUB_TOKEN = "github_pat_..."
   GITHUB_REPO = "machinelearning-source/UdeC-knowledge"
   GITHUB_BRANCH = "main"

   # Opcional: exige una clave compartida para publicar.
   # UPLOAD_PASSPHRASE = "clave-de-la-comunidad"
   ```

3. Guarda y reinicia la app.

### Seguridad

- El formulario es público: cualquier persona con el enlace puede publicar.
- Usa un token fine-grained limitado a este repositorio y solo con
  `Contents: Read and write`. No uses un token clásico con acceso total.
- Si hay spam, define `UPLOAD_PASSPHRASE` o elimina `GITHUB_TOKEN` para
  desactivar las subidas.
- El formulario limita a PDFs de máximo 15 MB, valida la firma del PDF, pide
  una verificación humana y aplica un límite de un envío por minuto por sesión.

## Alternativa sin token: Wiki editable por cualquiera

Si prefieres que la edición ocurra directamente en GitHub, activa la Wiki
pública:

1. Abre https://github.com/machinelearning-source/UdeC-knowledge/settings
2. En **Features** → **Wikis**, desmarca
   **Restrict editing to collaborators only**.

Desde ese momento cualquier usuario con cuenta de GitHub podrá editar la Wiki
del proyecto sin necesitar permisos de escritura sobre el código.

## Desarrollo local

```bash
pip install -r requirements.txt
python core/ingestion.py
streamlit run app.py
```
