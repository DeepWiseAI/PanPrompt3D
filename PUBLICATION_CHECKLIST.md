# Before public publication

- Review the full-model source, documentation and checkpoint manifests.
- Confirm the rights holder's license for PanPrompt3D-specific additions and
  trained weights. No new project-wide license has been chosen on their behalf.
- Verify upstream obligations and add any required attribution or license files.
- Confirm that training-data agreements permit distribution of trained weights.
- Publish code without data, local training output, credentials, or weight binaries
  in ordinary Git history; `.gitignore` excludes them.
- Upload only approved tensor-only weights as release assets, or opt into Git LFS.
- Do not force-add a resume checkpoint or local dataset.
- Run `python scripts/verify_release.py` after any changes.

No GitHub repository was created and no files were uploaded by this packaging step.
