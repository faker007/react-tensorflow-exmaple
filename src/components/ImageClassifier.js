import React from "react";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";

const ImageClassifier = () => {
  let net;
  const camera = React.useRef();
  const figures = React.useRef();
  const webcamElement = camera.current;

  const run = async () => {
    net = await mobilenet.load();

    const webcam = await tf.data.webcam(webcamElement, {
      resizeWidth: 870,
      resizeHeight: 534,
    });

    while (true) {
      const img = await webcam.capture();
      const result = await net.classify(img);

      if (figures.current) {
        figures.current.innerText = `예측: ${result[0].className} \n 예측 확률: ${result[0].probability}`;
      }

      img.dispose();

      await tf.nextFrame();
    }
  };

  React.useEffect(() => {
    run();
  });

  return (
    <>
      <div ref={figures}></div>
      <video
        autoPlay
        playsInline
        muted={true}
        ref={camera}
        width="870"
        height="534"
      />
    </>
  );
};

export default ImageClassifier;
