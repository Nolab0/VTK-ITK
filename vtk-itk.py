import itk
import matplotlib.pyplot as plt
import vtk

layer = 80
image = itk.imread("Data/case6_gre1.nrrd", itk.F)
image2 = itk.imread("Data/case6_gre2.nrrd", itk.F)

def Recalage(fixed, moving):

    fixed_image = itk.imread(fixed, itk.F)
    moving_image = itk.imread(moving, itk.F)

    type_ = type(fixed_image)
    outType_ = itk.Image[itk.F, 3]
    
    registration_method = itk.ImageRegistrationMethodv4[type_, type_].New()
    registration_method.SetFixedImage(fixed_image)
    registration_method.SetMovingImage(moving_image)
    
    transform_type = itk.TranslationTransform[itk.D, 3].New()
    transform_type.SetIdentity()
    registration_method.SetInitialTransform(transform_type)

    metric = itk.MeanSquaresImageToImageMetricv4[type_, type_].New()
    metric.SetFixedImage(fixed_image)
    metric.SetMovingImage(moving_image)

    registration_method.SetMetric(metric)

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
    optimizer.SetLearningRate(4.0)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(5)
    registration_method.SetOptimizer(optimizer)
    
    registration_method.Update()
    final_transform = registration_method.GetTransform()
    
    resampler = itk.ResampleImageFilter[type_, type_].New()
    resampler.SetInput(moving_image)
    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetTransform(final_transform)

    resampler.Update()
    resampled_image = resampler.GetOutput()

    cast_filter = itk.CastImageFilter[type_, outType_].New()
    cast_filter.SetInput(resampled_image)
    cast_filter.Update()
    return cast_filter.GetOutput()

recalee = Recalage("Data/case6_gre1.nrrd", "Data/case6_gre2.nrrd")

def segmentation(img, seedX, seedY, seedZ, lower, upper):
    type_ = type(img)
        
    filter_ = itk.GradientAnisotropicDiffusionImageFilter[type_, type_].New()
    filter_.SetInput(img) 
    output_image = filter_.GetOutput()

    connected_threshold_filter = itk.ConnectedThresholdImageFilter[type_, type_].New()
    connected_threshold_filter.SetInput(output_image)
    seed = itk.Index[3]()
    seed[0] = seedX
    seed[1] = seedY
    seed[2] = seedZ
    connected_threshold_filter.SetSeed(seed)
    connected_threshold_filter.Update()
    connected_threshold_filter.SetLower(lower)
    connected_threshold_filter.SetUpper(upper)
    connected_threshold_filter.SetReplaceValue(255)
    output_image = connected_threshold_filter.GetOutput()

    rescaleIntensityFilter = itk.RescaleIntensityImageFilter[type_, type_].New()
    rescaleIntensityFilter.SetInput(output_image)
    rescaleIntensityFilter.Update()

    return rescaleIntensityFilter.GetOutput()

segmented = segmentation(recalee, 125, 65, 80, 600, 800)
segmented2 = segmentation(image, 125, 65, 80, 600, 800)

itk.imwrite(segmented, "Data/segmented.nrrd")
itk.imwrite(segmented2, "Data/segmented2.nrrd")

# Create the first reader and set the file name
reader1 = vtk.vtkNrrdReader()
reader1.SetFileName('Data/segmented.nrrd')
reader1.Update()

# Create the second reader and set the file name
reader2 = vtk.vtkNrrdReader()
reader2.SetFileName('Data/segmented2.nrrd')
reader2.Update()

# Calculate the center of the volumes
(xMin, xMax, yMin, yMax, zMin, zMax) = reader1.GetExecutive().GetWholeExtent(reader1.GetOutputInformation(0))
(xSpacing, ySpacing, zSpacing) = reader1.GetOutput().GetSpacing()
(x0, y0, z0) = reader1.GetOutput().GetOrigin()
center1 = [x0 + xSpacing * 0.5 * (xMin + xMax),
           y0 + ySpacing * 0.5 * (yMin + yMax),
           z0 + zSpacing * 0.5 * (zMin + zMax)]

(xMin, xMax, yMin, yMax, zMin, zMax) = reader2.GetExecutive().GetWholeExtent(reader2.GetOutputInformation(0))
(xSpacing, ySpacing, zSpacing) = reader2.GetOutput().GetSpacing()
(x0, y0, z0) = reader2.GetOutput().GetOrigin()
center2 = [x0 + xSpacing * 0.5 * (xMin + xMax),
           y0 + ySpacing * 0.5 * (yMin + yMax),
           z0 + zSpacing * 0.5 * (zMin + zMax)]

# Matrices for axial, coronal, sagittal, oblique view orientations
axial = vtk.vtkMatrix4x4()
axial.DeepCopy((1, 0, 0, center1[0],
                0, 1, 0, center1[1],
                0, 0, 1, center1[2],
                0, 0, 0, 1))

coronal = vtk.vtkMatrix4x4()
coronal.DeepCopy((1, 0, 0, center1[0],
                  0, 0, 1, center1[1],
                  0, -1, 0, center1[2],
                  0, 0, 0, 1))

sagittal = vtk.vtkMatrix4x4()
sagittal.DeepCopy((0, 0, -1, center1[0],
                   1, 0, 0, center1[1],
                   0, -1, 0, center1[2],
                   0, 0, 0, 1))

oblique = vtk.vtkMatrix4x4()
oblique.DeepCopy((1, 0, 0, center1[0],
                  0, 0.866025, -0.5, center1[1],
                  0, 0.5, 0.866025, center1[2],
                  0, 0, 0, 1))

# Extract a slice in the desired orientation for the first file
reslice1 = vtk.vtkImageReslice()
reslice1.SetInputConnection(reader1.GetOutputPort())
reslice1.SetOutputDimensionality(2)
reslice1.SetResliceAxes(sagittal)
reslice1.SetInterpolationModeToLinear()

# Extract a slice in the desired orientation for the second file
reslice2 = vtk.vtkImageReslice()
reslice2.SetInputConnection(reader2.GetOutputPort())
reslice2.SetOutputDimensionality(2)
reslice2.SetResliceAxes(sagittal)
reslice2.SetInterpolationModeToLinear()

# Create a greyscale lookup table
table = vtk.vtkLookupTable()
table.SetRange(0, 2000)  # Image intensity range
table.SetValueRange(0.0, 1.0)  # From black to white
table.SetSaturationRange(0.0, 0.0)  # No color saturation
table.SetRampToLinear()
table.Build()

# Map the images through the lookup table
color1 = vtk.vtkImageMapToColors()
color1.SetLookupTable(table)
color1.SetInputConnection(reslice1.GetOutputPort())

color2 = vtk.vtkImageMapToColors()
color2.SetLookupTable(table)
color2.SetInputConnection(reslice2.GetOutputPort())

# Display the first image
actor1 = vtk.vtkImageActor()
actor1.GetMapper().SetInputConnection(color1.GetOutputPort())

# Display the second image
actor2 = vtk.vtkImageActor()
actor2.GetMapper().SetInputConnection(color2.GetOutputPort())

# Set up the renderers
renderer1 = vtk.vtkRenderer()
renderer1.AddActor(actor1)
renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)  # Display in the left half of the window

renderer2 = vtk.vtkRenderer()
renderer2.AddActor(actor2)
renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)  # Display in the right half of the window

# Set up the render window
window = vtk.vtkRenderWindow()
window.AddRenderer(renderer1)
window.AddRenderer(renderer2)
window.SetSize(800, 400)  # Set an appropriate size for the window

# Set up the interaction
interactorStyle = vtk.vtkInteractorStyleImage()
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(interactorStyle)
window.SetInteractor(interactor)
window.Render()

# Create callbacks for slicing the images
actions = {}
actions["Slicing"] = 0

def ButtonCallback(obj, event):
    if event == "LeftButtonPressEvent":
        actions["Slicing"] = 1
    else:
        actions["Slicing"] = 0

def MouseMoveCallback(obj, event):
    (lastX, lastY) = interactor.GetLastEventPosition()
    (mouseX, mouseY) = interactor.GetEventPosition()
    if actions["Slicing"] == 1:
        deltaY = mouseY - lastY
        reslice1.Update()
        reslice2.Update()
        sliceSpacing1 = reslice1.GetOutput().GetSpacing()[2]
        sliceSpacing2 = reslice2.GetOutput().GetSpacing()[2]
        matrix1 = reslice1.GetResliceAxes()
        matrix2 = reslice2.GetResliceAxes()
        # move the center point that we are slicing through for both images
        center1 = matrix1.MultiplyPoint((0, 0, sliceSpacing1 * deltaY, 1))
        center2 = matrix2.MultiplyPoint((0, 0, sliceSpacing2 * deltaY, 1))
        matrix1.SetElement(0, 3, center1[0])
        matrix1.SetElement(1, 3, center1[1])
        matrix1.SetElement(2, 3, center1[2])
        matrix2.SetElement(0, 3, center2[0])
        matrix2.SetElement(1, 3, center2[1])
        matrix2.SetElement(2, 3, center2[2])
        window.Render()
    else:
        interactorStyle.OnMouseMove()


interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)
interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)
interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)

# Start interaction
interactor.Start()
del renderer1
del renderer2
del window
del interactor
