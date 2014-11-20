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

	Mesh fish("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	fish.loadMesh("data/models/fish.obj");
	fish.setColorTexture("data/textures/fish.bmp", "myTextureSampler");
    fish.setModelMatrix(translate(fish.getModelMatrix(), vec3(0, 0, 0)));
	fish.setModelMatrix(scale(fish.getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));

	int num_fish = 10;
	vector<Mesh> v_fish;

	for( int i = 0 ; i < num_fish ; ++i){		
		Mesh e_fish("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
		v_fish.push_back(e_fish);
		v_fish[i].loadMesh("data/models/fish.obj");
		v_fish[i].setColorTexture("data/textures/fish.bmp", "myTextureSampler");
	    v_fish[i].setModelMatrix(translate(v_fish[i].getModelMatrix(), vec3(0, 0, 0)));
		v_fish[i].setModelMatrix(scale(v_fish[i].getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));
		
	}

	float speed = 30.0f;

	double lastTime = glfwGetTime();

	double coordx = 0;

	double r = 1;

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
		mat4 MVP = ProjectionMatrix * ViewMatrix * fish.getModelMatrix();

		
		// for( int i = 0 ; i < num_fish ; ++i){
		int i = 0;
			MVP = ProjectionMatrix * ViewMatrix * v_fish[i].getModelMatrix();
			v_fish[i].setModelMatrix(translate(v_fish[i].getModelMatrix(), vec3(-r*std::sin(coordx), (coordx), 0)));
			MVP = ProjectionMatrix * ViewMatrix * v_fish[i].getModelMatrix();
			v_fish[i].draw(MVP);
		// }

		//cout << std::sin(coordx) << " ";

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


