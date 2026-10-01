const copyButton = document.querySelector('#copy-bibtex');
const bibtex = document.querySelector('#bibtex');
const copyStatus = document.querySelector('#copy-status');

if (copyButton && bibtex && copyStatus) {
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent);
      copyStatus.textContent = 'BibTeX copied to the clipboard.';
      copyButton.textContent = 'Copied';
      window.setTimeout(() => {
        copyStatus.textContent = '';
        copyButton.textContent = 'Copy BibTeX';
      }, 2200);
    } catch (error) {
      copyStatus.textContent = 'Select the BibTeX block to copy it manually.';
    }
  });
}
