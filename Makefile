poetry:
	pip install cvxpy qpth tqdm tensorboard opencv-python torchvision Pillow
	# code --install-extension ms-python.python
	# rm ~/.vscode-server/bin/*/vscode-remote-lock*
	# code --install-extension equinusocio.vsc-community-material-theme
	chmod +x datasets/download_miniimagenet.sh
	./datasets/download_miniimagenet.sh
	rm miniimagenet.tar
	mv miniimagenet datasets/miniimagenet
	chmod +x download_trained_model.sh
	./download_trained_model.sh
	chmod +x download_pretrain_model.sh
	./download_pretrain_model.sh