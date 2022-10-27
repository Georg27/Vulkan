#version 450

layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
//	vec4 currentColor = 
	outFragColor = texture(samplerColor, inUV);//vec4(1.0f, 0.0f, 0.0f, 1.0f);
}