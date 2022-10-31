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
        const shortName = this.props.clsName.split(",")[0];
        return (
            <div>
                <div>
                    <div>
                        <strong>Name:</strong> {shortName}
                        <Definition clsIdx={this.props.classIdx} />
                    </div>
                </div>
                <strong>Sample Images:</strong>
                <div id="sample-images">{sampleImages}</div>
            </div>
        );
    }
}
