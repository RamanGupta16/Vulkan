#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
	// the values for fragColor will be automatically interpolated for the fragments between the three vertices, resulting in a smooth gradient.
    outColor = vec4(fragColor, 1.0);
}
