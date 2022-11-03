import React from "react";
import "./Instructions.css";

export default class Instructions extends React.Component {
    render() {
        return (
            <div id="instructions">
                <p>You will be shown a pair of images like this and asked two questions:</p>

                <img
                    alt=""
                    id="sample-img"
                    src="https://d3s-bucket.s3.amazonaws.com/d3s_samples/train/background-shift/112007.png"
                ></img>

                <h5>First Question</h5>
                <p>
                    As you can see, the image on the left has a background
                    (trees with snow) and a smaller image pasted on it (in the
                    above example it's some salmon, a type of fish). Ignore the
                    background. Just compare the object in the smaller pasted
                    image and the image on the right.{" "}
                    <strong>
                        If the image on the right has the same object as in the
                        pasted image, answer yes.{" "}
                    </strong>
                    In the above example, the right image also has salmon in it,
                    so you would answer yes in this case. Sometimes, the object
                    may be slightly hidden or out of shape. Answer yes even in
                    this case.{" "}
                    <strong>
                        Answer no only if the object is completely absent in the
                        right image.
                    </strong>
                </p>
                <p>
                    To help you identify this object, more information about the
                    object is shown in a panel on the right. This includes the
                    name of the object, its definition and 4 example images of
                    the object of interest.
                </p>
                <h5>Second Question</h5>
                <p>
                    Answer yes only if the right image has nudity or violence.
                    Note that this is pretty rare. Use your own judgement.
                </p>
            </div>
        );
    }
}
