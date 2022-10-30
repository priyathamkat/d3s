import Definition from "./Definition.js";
import React from "react";
import "./Metadata.css";

export default class Metadata extends React.Component {
    render() {
        const idxs = [0, 1, 2, 3];
        const sampleImages = idxs.map((idx) => (
            <img
                alt=""
                key={idx.toString()}
                src={
                    process.env.REACT_APP_BUCKET_URL +
                    "imagenet_samples/" +
                    this.props.classIdx +
                    "_" +
                    idx +
                    ".jpg"
                }
            ></img>
        ));
        return (
            <div>
                <div>
                    <div>
                        <h3>{this.props.clsName}</h3>
                        <Definition clsIdx={this.props.classIdx} />
                    </div>
                </div>
                <div id="sample-images">{sampleImages}</div>
            </div>
        );
    }
}
