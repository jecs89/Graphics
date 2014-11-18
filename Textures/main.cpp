#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <vector>
#include <algorithm>

#include <GL/glew.h>

#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
using namespace glm;
using namespace std;


#include "shader.h"
#include "controls.h"
#include "mesh.h"


int main(void)
{
	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	//window = glfwCreateWindow(1800, 1200, "Taller - Basics", NULL, NULL);
	window = glfwCreateWindow(1024, 768, "Taller - Basics", NULL, NULL);
	if (window == NULL){
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	glfwSetCursorPos(window, 1024 / 2, 768 / 2);
	glfwSwapInterval(1);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.1f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	vec3 lightPos = vec3(4, 8, 4);


	double k = 0.5;

	//Earth
	Mesh sphere("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	sphere.loadMesh("data/models/sphere.obj");
	sphere.setColorTexture("data/textures/colorEarth.png", "myTextureSampler");
	sphere.setModelMatrix(scale(sphere.getModelMatrix(), vec3(0.1*k, 0.1*k, 0.1*k)));
	// sphere.setModelMatrix(rotate(sphere.getModelMatrix(), -23.0f, vec3(0, 0, 1)));

	Mesh sun("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	sun.loadMesh("data/models/sphere.obj");
	sun.setColorTexture("data/textures/colorSun.png", "myTextureSampler");
	sun.setModelMatrix(scale(sun.getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));
/*
	Mesh sun2("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	sun2.loadMesh("data/models/sphere.obj");
	sun2.setColorTexture("data/textures/colorSun.png", "myTextureSampler");
	sun2.setModelMatrix(scale(sun2.getModelMatrix(), vec3(0.1, 0.1, 0.1)));

	Mesh sun3("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	sun3.loadMesh("data/models/sphere.obj");
	sun3.setColorTexture("data/textures/colorSun.png", "myTextureSampler");
	sun3.setModelMatrix(scale(sun3.getModelMatrix(), vec3(0.1, 0.1, 0.1)));

	Mesh sun4("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	sun4.loadMesh("data/models/sphere.obj");
	sun4.setColorTexture("data/textures/colorSun.png", "myTextureSampler");
	sun4.setModelMatrix(scale(sun4.getModelMatrix(), vec3(0.1, 0.1, 0.1)));
	sun4.setModelMatrix(translate(sun4.getModelMatrix(), vec3(2, 2, 2)));
*/	
	//Moon
	Mesh moon("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	moon.loadMesh("data/models/sphere.obj");
	moon.setColorTexture("data/textures/colorMoon.png", "myTextureSampler");
	moon.setModelMatrix(translate(moon.getModelMatrix(), vec3(0.05, 0, 0)));
	moon.setModelMatrix(scale(moon.getModelMatrix(), vec3(0.02*k, 0.02*k, 0.02*k)));

	float speed = 30.0f;


	double lastTime = glfwGetTime();

	double coordx = 0;
	double r = 0.1;
	double rearth = 0.06;
	double rmoon = 0.05;
	do
	{

		if (double(coordx) < 0.15) {
			coordx = 0;
		}
		coordx += 0.01;

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// time between two frames
		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		lastTime = currentTime;
		// Compute the MVP matrix from keyboard and mouse input
		//computeMatricesFromInputs();
		mat4 ProjectionMatrix = getProjectionMatrix();
		mat4 ViewMatrix = getViewMatrix();
		//Earth
		sphere.setModelMatrix(rotate(sphere.getModelMatrix(), speed*float(delta), vec3(0, 1, 0)));
		sphere.setModelMatrix(translate(sphere.getModelMatrix(), vec3( double(rearth*std::cos(coordx)), double(-rearth*std::sin(coordx)), 0)));
		mat4 MVP = ProjectionMatrix * ViewMatrix * sphere.getModelMatrix();
		sphere.draw(MVP);

		//Sun
		sun.setModelMatrix(rotate(sun.getModelMatrix(), speed*float(delta), vec3(0, 1, 0)));
		MVP = ProjectionMatrix * ViewMatrix * sun.getModelMatrix();
		sun.draw(MVP);

		/*sun2.setModelMatrix(rotate(sun2.getModelMatrix(), speed*float(delta), vec3(0, 1, 0)));
		sun2.setModelMatrix(translate(sun3.getModelMatrix(), vec3(0,0, 2)));
		MVP = ProjectionMatrix * ViewMatrix * sun2.getModelMatrix();
		sun2.draw(MVP);

		sun3.setModelMatrix(rotate(sun3.getModelMatrix(), speed*float(delta), vec3(0, 1, 0)));
		sun3.setModelMatrix(translate(sun3.getModelMatrix(), vec3(0,0,0)));
		MVP = ProjectionMatrix * ViewMatrix * sun3.getModelMatrix();
		sun3.draw(MVP);

		sun4.setModelMatrix(rotate(sun.getModelMatrix(), speed*float(delta), vec3(0, 1, 0)));
		sun4.setModelMatrix(translate(sun4.getModelMatrix(), vec3(2, 2, 2)));
		MVP = ProjectionMatrix * ViewMatrix * sun4.getModelMatrix();
		sun4.draw(MVP);
*/

		//Moon
		

		//cout << std::sin(coordx) << " ";
		moon.setModelMatrix(rotate(moon.getModelMatrix(), speed*float(delta), vec3(0, 1, 0)));
		moon.setModelMatrix(translate(moon.getModelMatrix(), vec3( double(rmoon*std::cos(coordx)) , double(-rmoon*std::sin(coordx)), 0)));
		MVP = ProjectionMatrix * ViewMatrix * moon.getModelMatrix();
		moon.draw(MVP);


		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
	glfwWindowShouldClose(window) == 0);



	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

