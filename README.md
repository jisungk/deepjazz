![deepjazz](https://jisungk.github.io/deepjazz/img/header_github.png)

### Using Keras & Theano for deep learning driven jazz generation

I built [*deepjazz*](http://deepjazz.io) in 36 hours for HackPrinceton, Spring 2016. It uses Keras & Theano, two deep learning libraries, to generate jazz music. Specifically, it builds a two-layer [LSTM](http://deeplearning.net/tutorial/lstm.html), learning from the given MIDI file. It uses deep learning, the AI tech that powers [Google's AlphaGo](https://deepmind.com/alpha-go.html) and [IBM's Watson](https://www.ibm.com/smarterplanet/us/en/ibmwatson/what-is-watson.html), **to make music -- something that's considered as deeply human**.

[![SoundCloud](https://jisungk.github.io/deepjazz/img/button_soundcloud.png)](https://soundcloud.com/deepjazz-ai)  
Check out deepjazz's music on **[SoundCloud](https://soundcloud.com/deepjazz-ai)**!

### Dependencies

* [Keras](http://keras.io/#installation)
* [Theano](http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions) ("bleeding-edge" version on GitHub)
* [music21](http://web.mit.edu/music21/doc/installing/index.html)

### Instructions

Run on CPU with command:  
```
python generator.py [# of epochs]
```

Run on GPU with command:  
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generator.py [# of epochs]
```

Note: `preprocess.py` must be modified to work with other MIDI files (the relevant "melody" MIDI part needs to be selected). The ability to handle this natively is a planned feature.

### Author

[Ji-Sung Kim](http://jisungkim.com)  
Princeton University, Department of Computer Science  
jisungk@princeton.edu  

### Citations

This project was inspired by and adapts a lot of preprocessing code (with permission) from Evan Chow's [jazzml](https://github.com/evancchow/jazzml). Thank you [Evan](https://www.linkedin.com/in/evancchow)! Public examples from the [Keras documentation](https://github.com/fchollet/keras) were also referenced.

### Code License, Media Copyright

Code is licensed under the Apache License 2.0  
Images and other media are copyrighted (Ji-Sung Kim)