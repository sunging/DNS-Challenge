from setuptools import setup, find_packages


setup(name="dnsmos",
      version="1.0.0",
      description="DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors",
      long_description="Human subjective evaluation is the ”gold standard” to evaluate speech quality optimized for human perception. Perceptual objective metrics serve as a proxy for subjective scores. The conventional and widely used metrics require a reference clean speech signal, which is unavailable in real recordings. The no-reference approaches correlate poorly with human ratings and are not widely adopted in the research community. One of the biggest use cases of these perceptual objective metrics is to evaluate noise suppression algorithms. DNSMOS generalizes well in challenging test conditions with a high correlation to human ratings in stack ranking noise suppression methods. More details can be found in [DNSMOS paper](https://arxiv.org/pdf/2010.15258.pdf).",
      author="sunging",
      url="https://github.com/sunging/DNS-Challenge",

      packages=find_packages(),
      package_data={
          '': ['DNSMOS/*.onnx']
      },
      install_requires=[
          'numpy>=1.22.4',
          'soundfile>=0.9.0',
          'librosa>=0.8.1',
          'pandas>=1.2.4',
          'onnxruntime>=1.13.1',
          'tqdm',
      ],
      python_requires='>=3.7',
      entry_points={
          'console_scripts': [
              'dnsmos = DNSMOS.dns_mos:main',
          ],
      },
      )
