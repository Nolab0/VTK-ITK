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

print("Recalage en cours...")
recalee = Recalage("Data/case6_gre1.nrrd", "Data/case6_gre2.nrrd")
print("Recalage terminé !")

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
    rescaleIntensityFilter.SetOutputMinimum(0)
    rescaleIntensityFilter.SetOutputMaximum(255)
    rescaleIntensityFilter.Update()

    out = itk.rescale_intensity_image_filter(output_image, output_minimum=0)

    return out

print("Segmentation en cours...")
segmented_before = segmentation(image, 125, 65, 80, 600, 800)
segmented_after = segmentation(recalee, 125, 65, 80, 600, 800)
print("Segmentation terminée !")

def compute_tumor_volume(image):
    image_type = type(image)
    statistics_filter = itk.StatisticsImageFilter[image_type].New()
    statistics_filter.SetInput(image)
    statistics_filter.Update()

    voxel_volume = image.GetSpacing()[0] * image.GetSpacing()[1] * image.GetSpacing()[2]
    tumor_volume = statistics_filter.GetSum() * voxel_volume

    return tumor_volume

tumor_volume_before = compute_tumor_volume(segmented_before)
tumor_volume_after = compute_tumor_volume(segmented_after)

print("Volume de la tumeur sur le scan 1: " + str(tumor_volume_before) + ' mm^3')
print("Volume de la tumeur sur le scan 2: " + str(tumor_volume_after) + ' mm^3')
print("Différence de volume entre les deux scans: " + str(tumor_volume_after - tumor_volume_before) + ' mm^3')

itk.imwrite(segmented_before, "Data/segmented.nrrd")
itk.imwrite(segmented_after, "Data/segmented2.nrrd")

reader1 = vtk.vtkNrrdReader()
reader1.SetFileName('Data/segmented.nrrd')
reader1.Update()

reader2 = vtk.vtkNrrdReader()
reader2.SetFileName('Data/segmented2.nrrd')
reader2.Update()

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

reslice1 = vtk.vtkImageReslice()
reslice1.SetInputConnection(reader1.GetOutputPort())
reslice1.SetOutputDimensionality(2)
reslice1.SetResliceAxes(sagittal)
reslice1.SetInterpolationModeToLinear()

reslice2 = vtk.vtkImageReslice()
reslice2.SetInputConnection(reader2.GetOutputPort())
reslice2.SetOutputDimensionality(2)
reslice2.SetResliceAxes(sagittal)
reslice2.SetInterpolationModeToLinear()

table = vtk.vtkLookupTable()
table.SetRange(0, 2000) 
table.SetValueRange(0.0, 1.0)
table.SetSaturationRange(0.0, 0.0)
table.SetRampToLinear()
table.Build()

color1 = vtk.vtkImageMapToColors()
color1.SetLookupTable(table)
color1.SetInputConnection(reslice1.GetOutputPort())

color2 = vtk.vtkImageMapToColors()
color2.SetLookupTable(table)
color2.SetInputConnection(reslice2.GetOutputPort())

actor1 = vtk.vtkImageActor()
actor1.GetMapper().SetInputConnection(color1.GetOutputPort())

actor2 = vtk.vtkImageActor()
actor2.GetMapper().SetInputConnection(color2.GetOutputPort())

renderer1 = vtk.vtkRenderer()
renderer1.AddActor(actor1)
renderer1.SetViewport(0.0, 0.0, 0.5, 1.0) 

renderer2 = vtk.vtkRenderer()
renderer2.AddActor(actor2)
renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer1)
window.AddRenderer(renderer2)
window.SetSize(800, 400)

interactorStyle = vtk.vtkInteractorStyleImage()
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(interactorStyle)
window.SetInteractor(interactor)
window.Render()

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

interactor.Start()
del renderer1
del renderer2
del window
del interactor

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, mapper1, mapper2, actor1, actor2):
        self.mapper1 = mapper1
        self.mapper2 = mapper2
        self.actor1 = actor1
        self.actor2 = actor2
        self.current_mapper = self.mapper1
        self.AddObserver("KeyPressEvent", self.onKeyPress)
        self.actor1.VisibilityOff()
        self.actor2.VisibilityOff()

    def onKeyPress(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        # Show the tumor before
        if key == "b":
            if self.current_mapper == self.mapper1:
                self.current_mapper = self.mapper2
                self.actor2.VisibilityOn()
            else:
                self.current_mapper = self.mapper1
                self.actor2.VisibilityOff()
            self.GetInteractor().Render()
        # Show the tumor after
        if key == "a":
            if self.current_mapper == self.mapper1:
                self.current_mapper = self.mapper2
                self.actor1.VisibilityOn()
            else:
                self.current_mapper = self.mapper1
                self.actor1.VisibilityOff()
            self.GetInteractor().Render()

head_scan = itk.vtk_image_from_image(itk.imread('Data/case6_gre1.nrrd'))
tumor_1_scan = itk.vtk_image_from_image(itk.imread('Data/segmented.nrrd'))
tumor_2_scan = itk.vtk_image_from_image(itk.imread('Data/segmented2.nrrd'))

# Create volume properties and mappers
mapper2 = vtk.vtkSmartVolumeMapper()
mapper2.SetInputData(head_scan)

opacity_transfer_function = vtk.vtkPiecewiseFunction()
opacity_transfer_function.AddPoint(0.0, 0.0)
opacity_transfer_function.AddPoint(90.0, 0.0)
opacity_transfer_function.AddPoint(100.0, 0.2)
opacity_transfer_function.AddPoint(120.0, 0.0)

colorFun = vtk.vtkColorTransferFunction()
colorFun.AddRGBPoint(0, 1, 1, 1)

volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(colorFun)
volume_property.SetScalarOpacity(opacity_transfer_function)

# Create volume actor and set transforms
volume = vtk.vtkVolume()
volume.SetMapper(mapper2)
volume.SetProperty(volume_property)

matrix = head_scan.GetPhysicalToIndexMatrix()
transf = vtk.vtkTransform()
transf.Translate(itk.dict_from_image(itk.imread('Data/case6_gre1.nrrd'))["origin"])
transf.PostMultiply()
transf.Concatenate(volume.GetMatrix())
volume.SetUserTransform(transf)

transf = vtk.vtkTransform()
transf.SetMatrix(matrix)
transf.PostMultiply()
transf.Concatenate(volume.GetMatrix())
volume.SetUserTransform(transf)

# Create contour filters, mappers, and actors for img[1] and img[2]
tumor1 = vtk.vtkContourFilter()
tumor1.SetInputData(tumor_1_scan)
tumor1.GenerateValues(1, 135, 135)

tumor2 = vtk.vtkContourFilter()
tumor2.SetInputData(tumor_2_scan)
tumor2.GenerateValues(1, 135, 135)

mapper1 = vtk.vtkDataSetMapper()
mapper1.SetInputConnection(tumor1.GetOutputPort())
mapper1.SetScalarVisibility(False)

actor1 = vtk.vtkActor()
actor1.SetMapper(mapper1)
actor1.GetProperty().SetColor(1, 0.5, 0)

mapper2 = vtk.vtkDataSetMapper()
mapper2.SetInputConnection(tumor2.GetOutputPort())
mapper2.SetScalarVisibility(False)

actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)
actor2.GetProperty().SetColor(1, 0, 0)
actor2.VisibilityOff()

# Create the renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderer.AddActor(actor1)
renderer.AddActor(actor2)
renderer.AddVolume(volume)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.Render()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Set the custom interactor style
custom_style = CustomInteractorStyle(mapper1, mapper2, actor1, actor2)
interactor.SetInteractorStyle(custom_style)

interactor.Start()
