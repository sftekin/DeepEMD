poetry:
	pip install cvxpy
	pip install qpth
	code --install-extension ms-python.python
	rm ~/.vscode-server/bin/*/vscode-remote-lock*
	code --install-extension equinusocio.vsc-community-material-theme
	chmod +x datasets/download_miniimagenet.sh
	./datasets/download_miniimagenet.sh
	rm miniimagenet.tar
	mv miniimagenet datasets/miniimagenet