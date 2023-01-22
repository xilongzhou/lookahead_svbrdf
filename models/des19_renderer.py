# import tensorflow as tf
import torch
from . import des19_helpers as helpers
import math
import numpy as np

Epsilon_tensor = torch.tensor(0.001).cuda()

#A renderer which implements the Cook-Torrance GGX rendering equations
class GGXRenderer:
    includeDiffuse = True

    def __init__(self, includeDiffuse = True):
        self.includeDiffuse = includeDiffuse

    #Compute the diffuse part of the equation
    def tf_diffuse(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi

    #Compute the diffuse part of the equation
    def torch_diffuse(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi

    #Compute the distribution function D driving the statistical orientation of the micro facets.
    def tf_D(self, roughness, NdotH):
        alpha = tf.square(roughness)
        underD = 1/tf.maximum(0.001, (tf.square(NdotH) * (tf.square(alpha) - 1.0) + 1.0))
        return (tf.square(alpha * underD)/math.pi)

    def torch_D(self, roughness, NdotH):
        alpha = torch.square(roughness)
        underD = 1/torch.max(Epsilon_tensor, (torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0))
        return (torch.square(alpha * underD)/math.pi)

    #Compute the fresnel approximation F
    def tf_F(self, specular, VdotH):
        sphg = tf.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
        return specular + (1.0 - specular) * sphg

    def torch_F(self, specular, VdotH):
        sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
        return specular + (1.0 - specular) * sphg


    #Compute the Geometry term (also called shadowing and masking term) G taking into account how microfacets can shadow each other.
    def tf_G(self, roughness, NdotL, NdotV):
        return self.G1(NdotL, tf.square(roughness)/2) * self.G1(NdotV, tf.square(roughness)/2)

    def G1(self, NdotW, k):
        return 1.0/tf.maximum((NdotW * (1.0 - k) + k), 0.001)


    def torch_G(self, roughness, NdotL, NdotV):
        return self.torch_G1(NdotL, torch.square(roughness)/2) * self.torch_G1(NdotV, torch.square(roughness)/2)

    def torch_G1(self, NdotW, k):
        return 1.0/torch.max((NdotW * (1.0 - k) + k), Epsilon_tensor)

    #This computes the equations of Cook-Torrance for a BRDF without taking light power etc... into account.
    def tf_calculateBRDF(self, svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight):

        h = helpers.tf_Normalize(tf.add(wiNorm, woNorm) / 2.0)

        ########################## des19 code #################################
        #Put all the parameter values between 0 and 1 except the normal as they should be used between -1 and 1 (as they express a direction in a 360° sphere)        
        diffuse = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,3:6]), 0.0,1.0), axis = 1)
        normals = tf.expand_dims(svbrdf[:,:,:,0:3], axis=1)
        normals = helpers.tf_Normalize(normals)
        specular = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,9:12]), 0.0, 1.0), axis = 1)
        roughness = tf.expand_dims(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,6:9]), 0.0, 1.0), axis = 1)
        #Avoid roughness = 0 to avoid division by 0
        roughness = tf.maximum(roughness, 0.001)

        #If we have multiple lights to render, add a dimension to handle it.
        if multiLight:
            diffuse = tf.expand_dims(diffuse, axis = 1)
            normals = tf.expand_dims(normals, axis = 1)
            specular = tf.expand_dims(specular, axis = 1)
            roughness = tf.expand_dims(roughness, axis = 1)

        NdotH = helpers.tf_DotProduct(normals, h)
        NdotL = helpers.tf_DotProduct(normals, wiNorm)
        NdotV = helpers.tf_DotProduct(normals, woNorm)

        VdotH = helpers.tf_DotProduct(woNorm, h)

        diffuse_rendered = self.tf_diffuse(diffuse, specular)
        D_rendered = self.tf_D(roughness, tf.maximum(0.0, NdotH))
        G_rendered = self.tf_G(roughness, tf.maximum(0.0, NdotL), tf.maximum(0.0, NdotV))
        F_rendered = self.tf_F(specular, tf.maximum(0.0, VdotH))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered
        
        #Add the diffuse part of the rendering if required.        
        if self.includeDiffuse:
            result = result + diffuse_rendered
        return result, NdotL


    #This computes the equations of Cook-Torrance for a BRDF without taking light power etc... into account.
    def torch_calculateBRDF(self, svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos):

        h = helpers.torch_Normalize((wiNorm+ woNorm) / 2.0)

        ########################## des19 code #################################
        #Put all the parameter values between 0 and 1 except the normal as they should be used between -1 and 1 (as they express a direction in a 360° sphere)        
        diffuse = torch.clamp(svbrdf[:,:,:,3:6], 0.0,1.0).unsqueeze(1)
        normals = svbrdf[:,:,:,0:3].unsqueeze(1)
        specular = torch.clamp(svbrdf[:,:,:,9:12], 0.0, 1.0).unsqueeze(1)
        roughness = torch.clamp(svbrdf[:,:,:,6:9], 0.0, 1.0).unsqueeze(1)
        #Avoid roughness = 0 to avoid division by 0
        roughness = torch.max(roughness, Epsilon_tensor)

        print('wiNorm.shape: ',wiNorm.shape)
        print('h.shape: ',h.shape)

        NdotH = helpers.torch_DotProduct(normals, h)
        print('roughness.shape: ',roughness.shape)
        print('NdotH.shape: ',NdotH.shape)

        NdotL = helpers.torch_DotProduct(normals, wiNorm)
        NdotV = helpers.torch_DotProduct(normals, woNorm)

        VdotH = helpers.torch_DotProduct(woNorm, h)

        diffuse_rendered = self.torch_diffuse(diffuse, specular)
        D_rendered = self.torch_D(roughness, torch.max(Epsilon_tensor, NdotH))
        G_rendered = self.torch_G(roughness, torch.max(Epsilon_tensor, NdotL), torch.max(Epsilon_tensor, NdotV))
        F_rendered = self.torch_F(specular, torch.max(Epsilon_tensor, VdotH))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered
        
        #Add the diffuse part of the rendering if required.        
        if self.includeDiffuse:
            result = result + diffuse_rendered

        print('result.shape: ',result.shape)
        return result, NdotL



    #Main rendering function, this is the computer graphics part, it generates an image from the parameter maps. For pure deep learning purposes the only important thing is that it is differentiable.
    def tf_Render(self, svbrdf, wi, wo, currentConeTargetPos, tensorboard = "", multiLight = False, currentLightPos = None, lossRendering = True, isAmbient = False, useAugmentation = True):
        print('before:',wi[0,0,:])
        wiNorm = helpers.torch_Normalize(wi)
        woNorm = helpers.torch_Normalize(wo)
        print('after:',wiNorm[0,0,:])

        #Calculate how the image should look like with completely neutral lighting
        result, NdotL = self.tf_calculateBRDF(svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight)
        resultShape = tf.shape(result)
        lampIntensity = 1.5
        
        #Add lighting effects
        if not currentConeTargetPos is None:
            #If we want a cone light (to have a flash fall off effect)
            currentConeTargetDir = currentLightPos - currentConeTargetPos #currentLightPos should never be None when currentConeTargetPos isn't
            coneTargetNorm = helpers.tf_Normalize(currentConeTargetDir)
            distanceToConeCenter = (tf.maximum(0.0, helpers.tf_DotProduct(wiNorm, coneTargetNorm)))
        if not lossRendering:
            #If we are not rendering for the loss
            if not isAmbient:
                if useAugmentation:
                    #The augmentations will allow different light power and exposures                 
                    stdDevWholeBatch = tf.exp(tf.random_normal((), mean = -2.0, stddev = 0.5))
                    #add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                    lampIntensity = tf.abs(tf.random_normal((resultShape[0], resultShape[1], 1, 1, 1), mean = 10.0, stddev = stdDevWholeBatch)) # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                    #autoExposure
                    autoExposure = tf.exp(tf.random_normal((), mean = np.log(1.5), stddev = 0.4))
                    lampIntensity = lampIntensity * autoExposure
                else:
                    lampIntensity = tf.reshape(tf.constant(13.0), [1, 1, 1, 1, 1]) #Look at the renderings when not using augmentations
            else:
                #If this uses ambient lighting we use much small light values to not burn everything.
                if useAugmentation:
                    lampIntensity = tf.exp(tf.random_normal((resultShape[0], 1, 1, 1, 1), mean = tf.log(0.15), stddev = 0.5)) #No need to make it change for each rendering.
                else:
                    lampIntensity = tf.reshape(tf.constant(0.15), [1, 1, 1, 1, 1])
            #Handle light white balance if we want to vary it..
            if useAugmentation and not isAmbient:
                whiteBalance = tf.abs(tf.random_normal([resultShape[0], resultShape[1], 1, 1, 3], mean = 1.0, stddev = 0.03))
                lampIntensity = lampIntensity * whiteBalance

            if multiLight:
                lampIntensity = tf.expand_dims(lampIntensity, axis = 2) #add a constant dim if using multiLight
        lampFactor = lampIntensity * math.pi

        if not isAmbient:
            if not lossRendering:
                #Take into accound the light distance (and the quadratic reduction of power)            
                lampDistance = tf.sqrt(tf.reduce_sum(tf.square(wi), axis = -1, keep_dims=True))
                lampFactor = lampFactor * helpers.tf_lampAttenuation_pbr(lampDistance)
            if not currentConeTargetPos is None:
                #Change the exponent randomly to simulate multiple flash fall off.            
                if useAugmentation:
                    exponent = tf.exp(tf.random_normal((), mean=np.log(5), stddev=0.35))
                else:
                    exponent = 5.0
                lampFactor = lampFactor * tf.pow(distanceToConeCenter, exponent)
                print("using the distance to cone center")

        result = result * lampFactor

        result = result * tf.maximum(0.0, NdotL)
        if multiLight:
            result = tf.reduce_sum(result, axis = 2) * 1.0#if we have multiple light we need to multiply this by (1/number of lights).
        if lossRendering:
            result = result / tf.expand_dims(tf.maximum(wiNorm[:,:,:,:,2], 0.001), axis=-1)# This division is to compensate for the cosinus distribution of the intensity in the rendering.

        return [result]#, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]


    #Main rendering function, this is the computer graphics part, it generates an image from the parameter maps. For pure deep learning purposes the only important thing is that it is differentiable.
    ## in our case, we do not consider varying intensity,
    def torch_Render(self, svbrdf, wi, wo, currentConeTargetPos, currentLightPos = None, lossRendering = True, isAmbient = False, useAugmentation = True):

        wiNorm = helpers.torch_Normalize(wi)
        woNorm = helpers.torch_Normalize(wo)

        #Calculate how the image should look like with completely neutral lighting
        result, NdotL = self.torch_calculateBRDF(svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos)
        resultShape = result.shape

        print('resultShape:',resultShape)

        # lampIntensity = 1.5
        # #Add lighting effects
        # if not currentConeTargetPos is None:
        #     #If we want a cone light (to have a flash fall off effect)
        #     currentConeTargetDir = currentLightPos - currentConeTargetPos #currentLightPos should never be None when currentConeTargetPos isn't
        #     coneTargetNorm = helpers.torch_Normalize(currentConeTargetDir)
        #     distanceToConeCenter = (torch.max(Epsilon_tensor, helpers.torch_DotProduct(wiNorm, coneTargetNorm)))

        ### we only use augmentation for ambient to render input images
        ### for flash light, we use fixed light intensity for optimization
        if not lossRendering:
            if not isAmbient:
                ## constant light intensity
                lampIntensity = torch.tensor(1.0)

                # if useAugmentation:
                #     #The augmentations will allow different light power and exposures                 
                #     # stdDevWholeBatch = tf.exp(tf.random_normal((), mean = -2.0, stddev = 0.5))
                #     stdDevWholeBatch = torch.exp(torch.normal(torch.tensor(-2.0), torch.tensor(0.5)))
                #     print('stdDevWholeBatch: ', stdDevWholeBatch)

                #     #add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                #     # lampIntensity = tf.abs(tf.random_normal((resultShape[0], resultShape[1], 1, 1, 1), mean = 10.0, stddev = stdDevWholeBatch)) # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                #     lampIntensity = torch.abs(torch.normal(mean=torch.tensor(10.0), std=stdDevWholeBatch)) 
                    
                #     #autoExposure
                #     # autoExposure = tf.exp(tf.random_normal((), mean = np.log(1.5), stddev = 0.4))
                #     autoExposure=torch.normal(torch.tensor(np.log(1.5)), torch.tensor(0.4))
                #     lampIntensity = lampIntensity * autoExposure

                #     print('lampIntensity: ', lampIntensity)
                # else:
                #     lampIntensity = tf.reshape(tf.constant(13.0), [1, 1, 1, 1, 1]) #Look at the renderings when not using augmentations
            else:
                #If this uses ambient lighting we use much small light values to not burn everything.
                if useAugmentation:
                    # lampIntensity = torch.exp(tf.random_normal((resultShape[0], 1, 1, 1, 1), mean = tf.log(0.15), stddev = 0.5)) #No need to make it change for each rendering.
                    temp_intensity=torch.normal(torch.tensor(np.log(0.15)), torch.tensor(0.5))
                    # m = tdist.Normal(torch.tensor([np.log(0.15)]),torch.tensor([0.5]))
                    # temp_intensity = m.sample((resultShape[0], 1, 1, 1, 1)).cuda().to(torch.float32)
                    print('temp_intensity: ',temp_intensity)

                    lampIntensity = torch.exp(temp_intensity)
                else:
                    lampIntensity = torch.reshape(torch.constant(0.15), [1, 1, 1, 1, 1])
                    lampIntensity = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.15])

            ## we do not need white balance
            #Handle light white balance if we want to vary it..
            # if useAugmentation and not isAmbient:
            #     # whiteBalance = tf.abs(tf.random_normal([resultShape[0], resultShape[1], 1, 1, 3], mean = 1.0, stddev = 0.03))
            #     whiteBalance = torch.abs(torch.normal(mean=torch.tensor(1.0), std=torch.tensor(0.03)))
            #     lampIntensity = lampIntensity * whiteBalance
            #     print('lampIntensity: ', lampIntensity)

            # if multiLight:
            #     lampIntensity = tf.expand_dims(lampIntensity, axis = 2) #add a constant dim if using multiLight
        
        lampFactor = lampIntensity * math.pi

        ### this is flash fall off
        ### we do not need this
        # if not isAmbient:
        #     if not lossRendering:
        #         #Take into accound the light distance (and the quadratic reduction of power)  
        #         # lampDistance = tf.sqrt(tf.reduce_sum(tf.square(wi), axis = -1, keep_dims=True))
        #         lampDistance = torch.sqrt(torch.sum(torch.square(wi),dim=-1, keepdim =True))
        #         print('lampFacto1: ',lampFactor)

        #         lampFactor = lampFactor * helpers.torch_lampAttenuation_pbr(lampDistance)
        #         print('lampFactor2: ',lampFactor)

        #     if not currentConeTargetPos is None:
        #         #Change the exponent randomly to simulate multiple flash fall off.            
        #         if useAugmentation:
        #             exponent = tf.exp(tf.random_normal((), mean=np.log(5), stddev=0.35))
        #             exponent = tf.exp(tf.random_normal((), mean=np.log(5), stddev=0.35))
        #         else:
        #             exponent = 5.0
        #         lampFactor = lampFactor * tf.pow(distanceToConeCenter, exponent)
        #         print("using the distance to cone center")


        print('lampFactor3: ', lampFactor)

        result = result * lampFactor.cuda()
        result = result * torch.max(Epsilon_tensor, NdotL)
        # if multiLight:
        #     result = tf.reduce_sum(result, axis = 2) * 1.0#if we have multiple light we need to multiply this by (1/number of lights).
        # if lossRendering:
        #     result = result / tf.expand_dims(tf.maximum(wiNorm[:,:,:,:,2], 0.001), axis=-1)# This division is to compensate for the cosinus distribution of the intensity in the rendering.

        return result