conda create -n sparsity python=3.9 -y
conda activate sparsity
conda install 'numpy<1.25' pytorch pytorch-cuda=11.8 boost psycopg2 -c pytorch -c nvidia -y
pip install cython zarr matplotlib mahotas scikit-learn
pip install git+https://github.com/funkelab/funlib.math.git
pip install git+https://github.com/funkelab/funlib.geometry.git@c2e21b921c70f71de161b9077b2e4ab11da796d9
pip install git+https://github.com/funkelab/funlib.persistence.git@84a9910e22647e75b3135b1cadfc219a379656ee
pip install git+https://github.com/funkelab/gunpowder.git@ecbb63c3ffbffa32db7534f6533a174265091699
pip install git+https://github.com/funkelab/daisy.git@d60ef4577c8de3aee61dd5eda2942df8db4b9457
pip install git+https://github.com/funkelab/funlib.learn.torch.git@5590fb51aef8381eeae99bbe75800ecb186684a1
pip install git+https://github.com/funkelab/funlib.segment.git@cdf607f0bde3b27f16cec39b9874e266be80bda9
pip install git+https://github.com/funkelab/funlib.evaluate.git@d2852b355910ee8081d0f6edc787c80cb94db909
pip install git+https://github.com/anforsm/lsd.git@781e6f830ec378565b5787d5008da71ca0bc2e62
pip install neuroglancer
pip install tensorboard tensorboardx

cd $CONDA_PREFIX/lib/python3.9/site-packages
git clone https://github.com/funkey/waterz.git
cd waterz
python setup.py install

cd