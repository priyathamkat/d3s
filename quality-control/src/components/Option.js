import React from "react";
import "./Option.css";

export default class Option extends React.Component {
    render() {
        return (
            <div>
                <input
                    className="form-check-input"
                    type={this.props.type}
                    name={this.props.name}
                    id={this.props.id}
                    value={this.props.option}
                    onChange={this.props.onChange}
                    checked={this.props.checked}
                    required={true}
                />
                <label className="form-check-label" htmlFor={this.props.id}>
                    {this.props.option}
                </label>
            </div>
        );
    }
}
