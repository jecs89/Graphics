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
	window = glfwCreateWindow(1900, 1000, "Taller - Basics", NULL, NULL);
	if (window == NULL){
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	glfwSetCursorPos(window, 1900 / 2, 1000 / 2);
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

	double k = 0.2;

	int n_fish = 4;
	vector<Mesh> v_fish(n_fish);

	for(int i = 0; i < n_fish; ++i){

		Mesh fish;
		fish.loadMesh("data/models/fish.obj");
		fish.setColorTexture("data/textures/fish.bmp", "myTextureSampler");
    	fish.setModelMatrix(translate(fish.getModelMatrix(), vec3(0.1*i, 0.1*i, 0)));
		fish.setModelMatrix(scale(fish.getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));

		fish.printModelMatrix();

		v_fish[i] = fish;

	}

	cout << "I'm alive\n";

	
	
	// Mesh* v_fish;

	// v_fish = (Mesh*) malloc( n_fish * sizeof(Mesh) );

	

	// Mesh fish2("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	// fish2.loadMesh("data/models/fish.obj");
	// fish2.setColorTexture("data/textures/fish.bmp", "myTextureSampler");
 //    fish2.setModelMatrix(translate(fish2.getModelMatrix(), vec3(0.1, 0.1, 0)));
	// fish2.setModelMatrix(scale(fish2.getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));

	// Mesh fish3("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	// fish3.loadMesh("data/models/fish.obj");
	// fish3.setColorTexture("data/textures/fish.bmp", "myTextureSampler");
 //    fish3.setModelMatrix(translate(fish3.getModelMatrix(), vec3(0.2, 0.2, 0)));
	// fish3.setModelMatrix(scale(fish3.getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));

	// Mesh fish4("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	// fish4.loadMesh("data/models/fish.obj");
	// fish4.setColorTexture("data/textures/fish.bmp", "myTextureSampler");
 //    fish4.setModelMatrix(translate(fish4.getModelMatrix(), vec3(0.3, 0.3, 0)));
	// fish4.setModelMatrix(scale(fish4.getModelMatrix(), vec3(0.3*k, 0.3*k, -1.2*k)));

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
		// computeMatricesFromInputs();
		mat4 ProjectionMatrix = getProjectionMatrix();
		mat4 ViewMatrix = getViewMatrix();
		// mat4 MVP = ProjectionMatrix * ViewMatrix * fish.getModelMatrix();

		// fish.setModelMatrix(translate(fish.getModelMatrix(), vec3(-r*std::sin(coordx), (coordx), 0)));
		// MVP = ProjectionMatrix * ViewMatrix * fish.getModelMatrix();
		// fish.draw(MVP);

		for( int i = 0; i < n_fish; ++i){
			v_fish[i].setModelMatrix(translate( v_fish[i].getModelMatrix(), vec3(-r*std::sin(coordx), (coordx), 0)));
			v_fish[i].printModelMatrix();
			mat4 MVP = ProjectionMatrix * ViewMatrix * v_fish[i].getModelMatrix();
			v_fish[i].draw(MVP);
		}

		// fish2.setModelMatrix(translate(fish2.getModelMatrix(), vec3(-r*std::sin(coordx), (coordx), 0)));
		// MVP = ProjectionMatrix * ViewMatrix * fish2.getModelMatrix();
		// fish2.draw(MVP);

		// fish3.setModelMatrix(translate(fish3.getModelMatrix(), vec3(-r*std::sin(coordx), (coordx), 0)));
		// MVP = ProjectionMatrix * ViewMatrix * fish3.getModelMatrix();
		// fish3.draw(MVP);

		// fish4.setModelMatrix(translate(fish4.getModelMatrix(), vec3(-r*std::sin(coordx), (coordx), 0)));
		// MVP = ProjectionMatrix * ViewMatrix * fish4.getModelMatrix();
		// fish4.draw(MVP);

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


